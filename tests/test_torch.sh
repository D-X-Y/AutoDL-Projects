# bash ./tests/test_torch.sh

pytest ./tests/test_torch_gpu_bugs.py::test_create -s
CUDA_VISIBLE_DEVICES="" pytest ./tests/test_torch_gpu_bugs.py::test_load -s
