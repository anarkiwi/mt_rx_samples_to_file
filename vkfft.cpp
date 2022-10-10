#include "vkFFT.h"
#include "utils_VkFFT.h"
#include "ShaderLang.h"
#include "sigpack/sigpack.h"

const uint64_t kVkNumBuf = 1;
static VkGPU vkGPU = {};
static VkFFTConfiguration vkConfiguration = {};
static VkDeviceMemory* vkBufferDeviceMemory = NULL;
static VkFFTApplication vkApp = {};
static VkFFTLaunchParams vkLaunchParams = {};


int64_t init_vkfft(size_t batches, size_t sample_id, size_t nfft) {
    vkGPU.enableValidationLayers = 0;

    VkResult res = VK_SUCCESS;
    res = createInstance(&vkGPU, sample_id);
    if (res) {
	std::cerr << "VKFFT_ERROR_FAILED_TO_CREATE_INSTANCE" << std::endl;
	return VKFFT_ERROR_FAILED_TO_CREATE_INSTANCE;
    }
    res = setupDebugMessenger(&vkGPU);
    if (res != 0) {
	//printf("Debug messenger creation failed, error code: %" PRIu64 "\n", res);
	return VKFFT_ERROR_FAILED_TO_SETUP_DEBUG_MESSENGER;
    }
    res = findPhysicalDevice(&vkGPU);
    if (res != 0) {
	//printf("Physical device not found, error code: %" PRIu64 "\n", res);
	return VKFFT_ERROR_FAILED_TO_FIND_PHYSICAL_DEVICE;
    }
    res = createDevice(&vkGPU, sample_id);
    if (res != 0) {
	//printf("Device creation failed, error code: %" PRIu64 "\n", res);
	return VKFFT_ERROR_FAILED_TO_CREATE_DEVICE;
    }
    res = createFence(&vkGPU);
    if (res != 0) {
	//printf("Fence creation failed, error code: %" PRIu64 "\n", res);
	return VKFFT_ERROR_FAILED_TO_CREATE_FENCE;
    }
    res = createCommandPool(&vkGPU);
    if (res != 0) {
	//printf("Fence creation failed, error code: %" PRIu64 "\n", res);
	return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_POOL;
    }
    vkGetPhysicalDeviceProperties(vkGPU.physicalDevice, &vkGPU.physicalDeviceProperties);
    vkGetPhysicalDeviceMemoryProperties(vkGPU.physicalDevice, &vkGPU.physicalDeviceMemoryProperties);

    glslang_initialize_process();//compiler can be initialized before VkFFT

    vkConfiguration.FFTdim = 1;
    vkConfiguration.size[0] = nfft;
    vkConfiguration.size[1] = 1;
    vkConfiguration.size[2] = 1;
    vkConfiguration.device = &vkGPU.device;
    vkConfiguration.queue = &vkGPU.queue;
    vkConfiguration.fence = &vkGPU.fence;
    vkConfiguration.commandPool = &vkGPU.commandPool;
    vkConfiguration.physicalDevice = &vkGPU.physicalDevice;
    vkConfiguration.isCompilerInitialized = true;
    vkConfiguration.doublePrecision = false;
    vkConfiguration.numberBatches = batches;

    std::cerr << "using vkFFT batch size " << vkConfiguration.numberBatches << " on " << vkGPU.physicalDeviceProperties.deviceName << std::endl;

    vkConfiguration.bufferSize = (uint64_t*)aligned_alloc(sizeof(uint64_t), sizeof(uint64_t) * kVkNumBuf);
    if (!vkConfiguration.bufferSize) return VKFFT_ERROR_MALLOC_FAILED;

    const size_t buffer_size_f = vkConfiguration.size[0] * vkConfiguration.size[1] * vkConfiguration.size[2] * vkConfiguration.numberBatches / kVkNumBuf;
    for (uint64_t i = 0; i < kVkNumBuf; ++i) {
	vkConfiguration.bufferSize[i] = (uint64_t)sizeof(std::complex<float>) * buffer_size_f;
    }

    vkConfiguration.bufferNum = kVkNumBuf;

    vkConfiguration.buffer = (VkBuffer*)aligned_alloc(sizeof(std::complex<float>), kVkNumBuf * sizeof(VkBuffer));
    if (!vkConfiguration.buffer) return VKFFT_ERROR_MALLOC_FAILED;
    vkBufferDeviceMemory = (VkDeviceMemory*)aligned_alloc(sizeof(std::complex<float>), kVkNumBuf * sizeof(VkDeviceMemory));
    if (!vkBufferDeviceMemory) return VKFFT_ERROR_MALLOC_FAILED;

    for (uint64_t i = 0; i < kVkNumBuf; ++i) {
	vkConfiguration.buffer[i] = {};
	vkBufferDeviceMemory[i] = {};
	VkFFTResult resFFT = allocateBuffer(&vkGPU, &vkConfiguration.buffer[i], &vkBufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, vkConfiguration.bufferSize[i]);
	if (resFFT != VKFFT_SUCCESS) return VKFFT_ERROR_MALLOC_FAILED;
    }

    vkLaunchParams.buffer = vkConfiguration.buffer;

    VkFFTResult resFFT = initializeVkFFT(&vkApp, vkConfiguration);
    if (resFFT != VKFFT_SUCCESS) return resFFT;

    return VKFFT_SUCCESS;
}


void free_vkfft() {
    for (uint64_t i = 0; i < kVkNumBuf; i++) {
	vkDestroyBuffer(vkGPU.device, vkConfiguration.buffer[i], NULL);
	vkFreeMemory(vkGPU.device, vkBufferDeviceMemory[i], NULL);
    }
    free(vkConfiguration.buffer);
    free(vkBufferDeviceMemory);
    free(vkConfiguration.bufferSize);
    deleteVkFFT(&vkApp);
    vkDestroyFence(vkGPU.device, vkGPU.fence, NULL);
    vkDestroyCommandPool(vkGPU.device, vkGPU.commandPool, NULL);
    vkDestroyDevice(vkGPU.device, NULL);
    DestroyDebugUtilsMessengerEXT(&vkGPU, NULL);
    vkDestroyInstance(vkGPU.instance, NULL);
    glslang_finalize_process();
}


void vkfft_specgram_offload(arma::cx_fmat &Pw_in, arma::cx_fmat &Pw) {
    size_t row_size = Pw_in.n_rows * sizeof(std::complex<float>);

    // TODO: offload windowing.
    for(arma::uword k=0; k < Pw_in.n_cols; k += vkConfiguration.numberBatches)
    {
	size_t offset = k * row_size;
	// TODO: retain overlap buffer from previous spectrum() call.
	uint64_t buffer_size = std::min(vkConfiguration.bufferSize[0], (uint64_t)((Pw_in.n_cols - k) * row_size));
	transferDataFromCPU(&vkGPU, ((char*)Pw_in.memptr()) + offset, &vkConfiguration.buffer[0], buffer_size);
	performVulkanFFT(&vkGPU, &vkApp, &vkLaunchParams, -1, 1);
	transferDataToCPU(&vkGPU, ((char*)Pw.memptr()) + offset, &vkConfiguration.buffer[0], buffer_size);
    }
}
