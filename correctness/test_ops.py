import unittest

import torch

from experiments.profile_ops import check_case, make_cases
from kernels.ops import load_ops


class TorchBaselineTests(unittest.TestCase):
    def test_baselines_against_double_reference(self):
        torch.manual_seed(123)
        for shape in [(0,), (1,), (257,), (17, 31)]:
            x = torch.randn(shape, dtype=torch.complex64)
            y = torch.randn_like(x)
            for op in ("copy", "add", "multiply"):
                xd, yd = x.to(torch.complex128), y.to(torch.complex128)
                reference = xd if op == "copy" else xd + yd if op == "add" else xd * yd
                for name, run, outputs in make_cases(torch, None, x, y, op):
                    if name.startswith("torch"):
                        with self.subTest(shape=shape, op=op, case=name):
                            check_case(torch, run, outputs, reference, op)


@unittest.skipUnless(torch.cuda.is_available(), "requires an NVIDIA CUDA device")
class CUDAKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extension = load_ops()

    def test_kernels_on_nondefault_stream(self):
        # Includes empty inputs, tails, vectors, matrices, and nonzero storage offsets.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), torch.inference_mode():
            for shape in [(0,), (1,), (255,), (256,), (257,), (17, 31)]:
                x = torch.randn(shape, device="cuda", dtype=torch.complex64)
                y = torch.randn_like(x)
                if len(shape) == 1 and shape[0]:
                    x = torch.cat([x[:1], x])[1:]
                for op in ("copy", "add", "multiply"):
                    xd, yd = x.to(torch.complex128), y.to(torch.complex128)
                    reference = xd if op == "copy" else xd + yd if op == "add" else xd * yd
                    for name, run, outputs in make_cases(torch, self.extension, x, y, op):
                        with self.subTest(shape=shape, op=op, case=name):
                            check_case(torch, run, outputs, reference, op)
        stream.synchronize()

    def test_rejects_invalid_inputs(self):
        x = torch.ones(8, device="cuda", dtype=torch.complex64)
        with self.assertRaises(RuntimeError):
            self.extension.run(0, False, [x], [x])
        with self.assertRaises(RuntimeError):
            self.extension.run(0, False, [x[::2]], [torch.empty(4, device="cuda", dtype=x.dtype)])
        with self.assertRaises(RuntimeError):
            self.extension.run(0, False, [x.conj()], [torch.empty_like(x)])


if __name__ == "__main__":
    unittest.main()
