#include <torch/extension.h>
#include <optional>
#include <vector>

std::vector<torch::Tensor> swiglu_cuda(
    torch::Tensor y, std::optional<torch::Tensor> g,
    std::optional<torch::Tensor> bias, std::optional<torch::Tensor> weights);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu_cuda", &swiglu_cuda, "Fused SwiGLU CUDA C forward and backward");
}
