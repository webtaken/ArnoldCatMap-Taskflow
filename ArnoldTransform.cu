/*
Para compilar el siguiente proyecto colocar el siguiente comando:
nvcc -std=c++17 -I <path to taskflow repo> --extended-lambda --gpu-architecture=compute_<your compute GPU's capability> ArnoldTransform.cu -o ArnoldTransform
*/
// TASKFLOW
#include <taskflow/taskflow.hpp>      // core taskflow routines
#include <taskflow/cuda/cudaflow.hpp> // core cudaflow routines

// STB_IMAGE
#include "stb_image/stb_image.cpp"
#include "stb_image/stb_image_write.cpp"

// STD C++
#include <iostream>
#include <ostream>

class IMG_PROPS
{
public:
    int width, height, no_channels, desired_no_channels;
    IMG_PROPS(int _width, int _height, int _no_channels, int _desired_no_channels){
        width = _width;
        height = _height;
        no_channels = _no_channels;
        desired_no_channels = _desired_no_channels;
    }
    ~IMG_PROPS(){}
};


// Arnold's Cat Map Kernel
__global__ void ArnoldTransformKernel(unsigned char* Pin, unsigned char* Pout, int N, int channels)
{
    int Col = threadIdx.x + blockDim.x * blockIdx.x;
    int Row = threadIdx.y + blockDim.y * blockIdx.y;

    if(Col < N && Row < N){
        // Ver el siguient enlace para ver la fórmula de la transformada de Arnold
        // http://fibonacci.math.uri.edu/~kulenm/diffeqaturi/victor442/index.html
        int newCol = (Col + Row) % N;
        int newRow = (Col + 2*Row) % N;
        
        int offset = (Row * N + Col) * channels;
        int newOffset = (newRow * N + newCol) * channels;

        // Valores RGB 
        Pout[newOffset] = Pin[offset]; // R
        Pout[newOffset + 1] = Pin[offset + 1]; // G
        Pout[newOffset + 2] = Pin[offset + 2]; // B
    }
}

// main function begins
int main(int argc, char **argv)
{
    // se debe proveer el path de la imagen a recibir
    if(argc < 2){
        std::cout << "You must provide and image name\n";
        exit(EXIT_FAILURE);
    }
    // en el primer argumento se lee el nombre del archivo
    std::string img_path(argv[1]);
    std::string img_filename = img_path.substr(0, img_path.find('.'));
    std::string img_extention = img_path.substr(img_path.find('.'));
    
    // if(img_extention != ".jpg"){
    //     std::cout << "Only .jpg images are allowed\n";
    //     exit(EXIT_FAILURE);
    // }

    int width, height, original_no_channels;
    int desired_no_channels = 3; // solo 3 porque solo procesaremos imágenes .jpg
    unsigned char *img = stbi_load(img_path.c_str(), &width, &height,
                                   &original_no_channels, desired_no_channels);
    if (img == NULL)
    {
        printf("Error in loading the image\n");
        exit(EXIT_FAILURE);
    }
    IMG_PROPS img_props(width, height, original_no_channels, desired_no_channels);
    printf("Loaded image characteristics:\n");
    printf("width: %dpx\n", width); 
    printf("height: %dpx\n", height);
    printf("original N channels: %d\n", original_no_channels); 
    printf("loaded with N channels: %d\n", desired_no_channels);

    tf::Taskflow taskflow;
    tf::Executor executor;

    taskflow.name("Arnold's Cat Map Algorithm");

/////////////////////////////////////////////////////////////////////////////////////////////
    // la iteración N-ésima de todo el proceso y el máximo número de iteraciones
    int iter_transform = 0, max_iters = 1000; 

    unsigned char *h_Pin = {nullptr}; // imagen adaptada a dimensión NxN
    unsigned char *h_Pout = {nullptr}; // imagen adaptada resultante de la transformada

    // Imágenes a ser alojadas en la GPU
    unsigned char *d_Pin = {nullptr}; // imagen adaptada en la GPU
    unsigned char *d_Pout = {nullptr}; // imagen resultante de la transformación en la GPU
    bool init1 = true, init2 = true; // flags para una sola reserva en memoria en GPU de d_Pin y d_Pout

    // tamaño de la imagen adaptada NxN con sus respectivos canales
    int N = std::max(width, height);
////////////////////////////////////////////////////////////////////////////////////////////

    // aquí colocaremos las funciones de taskflow
    tf::Task resizer = taskflow.emplace([&]()
    {
        // la imagen adaptada es de NxNxChannels
        h_Pin = new unsigned char[N * N * img_props.desired_no_channels];
        // reservamos memoria para el resultado de la transformación continua
        h_Pout = new unsigned char[N * N * img_props.desired_no_channels];

        if(img_props.width > img_props.height)
        {
            // esta conversión es mas sencilla porque la imagen está en row order
            int i;
            // copiamos la imagen tal cual en la imagen adaptada
            for (i = 0; i < img_props.width * img_props.height * img_props.desired_no_channels; i++)
                h_Pin[i] = img[i];
            // rellenamos el resto de pixeles negros (0s) el resto de la imagen adaptada
            for(; i < N * N * img_props.desired_no_channels; i++)
                h_Pin[i] = (unsigned char)0; // rellenamos con un pixel negro
        }
        else // width <= height
        {
            for (int j = 0; j < N; j++)
            {
                int h = 0;
                for (int i = 0; i < N; i++)
                {
                    int offset = (j * N + i) * img_props.desired_no_channels;
                    int offset_orig = (j * img_props.width + h) * img_props.desired_no_channels;
                    if (offset < ((j * N + img_props.width) * img_props.desired_no_channels))
                    {
                        h_Pin[offset] = img[offset_orig]; // R
                        h_Pin[offset + 1] = img[offset_orig + 1]; // G
                        h_Pin[offset + 2] = img[offset_orig + 2]; // B
                        h++;
                    }
                    else
                    {
                        h_Pin[offset] = (unsigned char)0; // rellenamos con un pixel negro
                    }
                }
            }
        }

        std::string transformed_image_name = img_filename + "_arnold_iter_" 
                                        + std::to_string(iter_transform) + img_extention;
        iter_transform++; // incrementamos la iteración
        // escribimos la imagen resultante en la misma carpeta
        stbi_write_jpg(transformed_image_name.c_str(), img_props.width, img_props.height, 
                                    img_props.no_channels, h_Pin, 100);
    }).name("resizer");

    // una tarea puente
    tf::Task helper = taskflow.emplace(
        [&](){}).name("Helper");

    auto [alloc_Pin, alloc_Pout] = taskflow.emplace(
        [&]()
        { 
            if (init1)
            {
                cudaMalloc(&d_Pin, N * N * img_props.desired_no_channels * sizeof(unsigned char)); 
                init1 = false;
            }
        },
        [&]()
        { 
            if(init2) {
                cudaMalloc(&d_Pout, N * N * img_props.desired_no_channels * sizeof(unsigned char));
                init2 = false;
            }
        }
    );
    // colocando el nombre de cada proceso de alojación
    alloc_Pin.name("alloc_Pin");
    alloc_Pout.name("alloc_Pout");

    tf::Task arnoldflow = taskflow.emplace([&](tf::cudaFlow &cf) {
        // transfiriendo datos de la imagen adaptada (host) a la memoria en el dispositivo
        tf::cudaTask Pin_h2d = cf.copy(d_Pin, h_Pin, N * N * img_props.desired_no_channels).name("Pin_h2d");
        // transfiriendo el resultado del dispositivo al host
        tf::cudaTask Pout_d2h = cf.copy(h_Pout, d_Pout, N * N * img_props.desired_no_channels).name("Pout_d2h");

        // dimensiones del grid
        dim3 dimGrid(ceil(N / 16.0f), ceil(N / 16.0f), 1);
        // dimensiones de los bloques
        dim3 dimBlock(16, 16, 1);

        // launch ArnoldKernel<<<dimGrid, dimBlock, 0>>>(d_Pin, d_Pout, N, 3)
        tf::cudaTask ArnoldKernel = cf.kernel(
            dimGrid, dimBlock, 0, ArnoldTransformKernel, d_Pin, d_Pout, N, 
            img_props.desired_no_channels).name("ArnoldKernel");

        // construimos el flujo de trabajo
        ArnoldKernel.succeed(Pin_h2d).precede(Pout_d2h);
    }).name("ArnoldTransformCudaFlow");

    // esta es una función condicional
    tf::Task convergence_checker = taskflow.emplace([&]() {
        // se llego al tope de las iteraciones
        if(iter_transform >= max_iters){
            // En este caso se ha llegado a la imagen original, entonces pasamos a la tarea final
            // Antes liberamos la memoria reservada en la GPU de las dos imágenes pasadas
            cudaFree(d_Pin);
            cudaFree(d_Pout);
            return 1;     
        }

        // compararemos la imagen original img con el resultado de la transformación h_Pout
        for (int j = 0, y = 0; j < img_props.height; j++, y++)
        {
            for (int i = 0; i < img_props.width; i++)
            {
                int offset_orig = (j * img_props.width + i) * img_props.desired_no_channels;
                int offset_transform = (y * N + i) * img_props.desired_no_channels;
                if (h_Pout[offset_transform] != img[offset_orig] ||         // R
                    h_Pout[offset_transform + 1] != img[offset_orig + 1] || // G
                    h_Pout[offset_transform + 2] != img[offset_orig + 2]) // B
                {
                    // copiamos los datos de h_Pout hacia h_Pin para la siguiente iteración
                    memcpy(h_Pin, h_Pout, N * N * img_props.desired_no_channels * sizeof(unsigned char));
                    
                    // antes guardaremos la imagen de la iteración N ésima del proceso
                    // de la transformada 
                    unsigned char *tmp_img = new unsigned char[img_props.width * img_props.height 
                                                        * img_props.desired_no_channels];

                    for(int p = 0; p < img_props.height; p++){
                        for(int q = 0; q < img_props.width; q++){
                            int offset_orig = (p * img_props.width + q) * img_props.desired_no_channels;
                            int offset_transform = (p * N + q) * img_props.desired_no_channels;
                
                            // copiando los valores RGB en la imagen a guardarse
                            tmp_img[offset_orig] = h_Pin[offset_transform]; // R
                            tmp_img[offset_orig + 1] = h_Pin[offset_transform + 1]; // G
                            tmp_img[offset_orig + 2] = h_Pin[offset_transform + 2]; // B
                        }
                    }

                    std::string transformed_image_name = img_filename + "_arnold_iter_" 
                                                        + std::to_string(iter_transform) + img_extention;
                    iter_transform++; // incrementamos la iteración
                    // escribimos la imagen resultante en la misma carpeta
                    stbi_write_jpg(transformed_image_name.c_str(), img_props.width, img_props.height, 
                                    img_props.no_channels, tmp_img, 100);
                    
                    // siempre debemos liberar la memoria
                    delete [] tmp_img;
                    // volvemos a la primera función
                    return 0;
                }
            }
        }

        // copiamos los datos de h_Pout hacia h_Pin para copiar el resultado final
        memcpy(h_Pin, h_Pout, N * N * img_props.desired_no_channels * sizeof(unsigned char));
        // antes guardaremos la última imagen de la iteración N ésima del proceso
        // de la transformada de Arnold
        unsigned char *tmp_img = new unsigned char[img_props.width * img_props.height 
                                                        * img_props.desired_no_channels];

        for(int p = 0; p < img_props.height; p++){
            for(int q = 0; q < img_props.width; q++){
                int offset_orig = (p * img_props.width + q) * img_props.desired_no_channels;
                int offset_transform = (p * N + q) * img_props.desired_no_channels;
                
                // copiando los valores RGB en la imagen a guardarse
                tmp_img[offset_orig] = h_Pin[offset_transform]; // R
                tmp_img[offset_orig + 1] = h_Pin[offset_transform + 1]; // G
                tmp_img[offset_orig + 2] = h_Pin[offset_transform + 2]; // B
            }
        }
                    
        std::string transformed_image_name = img_filename + "_arnold_iter_" 
                                    + std::to_string(iter_transform) + img_extention;
        // escribimos la imagen resultante en la misma carpeta
        stbi_write_jpg(transformed_image_name.c_str(), img_props.width, img_props.height, 
                                img_props.no_channels, tmp_img, 100);

        // volvemos a la primera función
        // En este caso se ha llegado a la imagen original, entonces pasamos a la tarea final
        // Antes liberamos la memoria reservada en la GPU de las dos imágenes pasadas
        cudaFree(d_Pin);
        cudaFree(d_Pout);
        // siempre debemos liberar la memoria
        delete [] tmp_img;
        
        return 1;     
    }).name("convergence&cls");



    // La última tarea a realizarse
    tf::Task finalizer = taskflow.emplace(
        [&]()
        { 
            std::cout << "Arnold Transformation ended\n";
            std::cout << "Image path: " << img_path << "\n";
            std::cout << "With " << iter_transform << " iterations\n";

            // Terminamos las tareas liberando los recursos empleados
            // siempre debemos liberar la memoria
            stbi_image_free(img);

            // liberamos la memoria de ambas imágenes en CPU
            if (h_Pin != nullptr)
            {
                delete[] h_Pin; // también debemos liberar esta memoria
            }
            if (h_Pout != nullptr)
            {
                delete[] h_Pout; // también debemos liberar esta memoria
            }
        }).name("Finalizer");

//////////////////////////////////////////////////////////////////////////////////////////
    // Ahora construimos el grafo de tareas
    resizer.precede(helper);
    helper.precede(alloc_Pin, alloc_Pout);
    arnoldflow.succeed(alloc_Pin, alloc_Pout);
    arnoldflow.precede(convergence_checker);
    convergence_checker.precede(helper, finalizer);
    
    executor.run(taskflow).wait();

    taskflow.dump(std::cout); // mostramos el grafo como un archivo de graphviz

}