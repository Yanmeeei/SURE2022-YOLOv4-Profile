all testing results move to ./results directory. 

##profilerWrapper
A wrapper that wraps all profile tools including timer, memorizer, data_scale and dependency_recorder.

##How to use
For each layer, wrap it in the following code block:
```python
prof_wrapper.scale.dependency_check(tensor_name="x2", src="d1_conv2", dest="d1_conv4")
tmp_input = torch.clone(x2)
with profile(
        activities=
        [
            ProfilerActivity.CPU
        ] if not usingcuda else
        [
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA
        ],
        profile_memory=True, record_shapes=True
) as prof:
    with record_function("model_inference"):
        self.conv4(tmp_input)
prof_report = str(prof.key_averages().table()).split("\n")
prof_wrapper.mr.get_mem("d1_conv4", prof_report, usingcuda)

prof_wrapper.tt.tic("d1_conv4")
x4 = self.conv4(x2)
prof_wrapper.tt.toc("d1_conv4")
prof_wrapper.scale.weight(tensor_src="d1_conv4", data=x4)
```

