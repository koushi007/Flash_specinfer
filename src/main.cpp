#include <torch/extension.h>

std::vector<torch::Tensor> forward_1(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor StartTimes, torch::Tensor EndTimes, bool IsTree
);
std::vector<torch::Tensor> forward_2(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor StartTimes, torch::Tensor EndTimes, bool IsTree
);
std::vector<torch::Tensor> forward_3(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor StartTimes, torch::Tensor EndTimes, bool IsTree
);
std::vector<torch::Tensor> forward_4(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor StartTimes, torch::Tensor EndTimes, bool IsTree
);
std::vector<torch::Tensor> forward_5(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor StartTimes, torch::Tensor EndTimes, bool IsTree
);
std::vector<torch::Tensor> forward_6(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor StartTimes, torch::Tensor EndTimes, bool IsTree
);
std::vector<torch::Tensor> forward_7(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor StartTimes, torch::Tensor EndTimes, bool IsTree
);
std::vector<torch::Tensor> forward_8(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor StartTimes, torch::Tensor EndTimes, bool IsTree
);


std::vector<torch::Tensor> forward_base_2(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor CausalMask, bool IsTree
);
std::vector<torch::Tensor> forward_base_3(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor CausalMask, bool IsTree
);
std::vector<torch::Tensor> forward_base_4(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor CausalMask, bool IsTree
);
std::vector<torch::Tensor> forward_base_5(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor CausalMask, bool IsTree
);
std::vector<torch::Tensor> forward_base_6(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor CausalMask, bool IsTree
);
std::vector<torch::Tensor> forward_base_7(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor CausalMask, bool IsTree
);
std::vector<torch::Tensor> forward_base_8(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor CausalMask, bool IsTree
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_1", &forward_1);
    m.def("forward_2", &forward_2);
    m.def("forward_3", &forward_3);
    m.def("forward_4", &forward_4);
    m.def("forward_5", &forward_5);
    m.def("forward_6", &forward_6);
    m.def("forward_7", &forward_7);
    m.def("forward_8", &forward_8);
    
    m.def("forward_base_2", &forward_base_2);
    m.def("forward_base_3", &forward_base_3);
    m.def("forward_base_4", &forward_base_4);
    m.def("forward_base_5", &forward_base_5);
    m.def("forward_base_6", &forward_base_6);
    m.def("forward_base_7", &forward_base_7);
    m.def("forward_base_8", &forward_base_8);
}
