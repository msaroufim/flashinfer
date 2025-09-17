# NVRTC Integration Evaluation for FlashInfer

## Summary
Implemented PyTorch `_compile_kernel` (NVRTC) backend as alternative to ninja-based compilation in FlashInfer's JIT system.

## Performance Results
- **Simple kernels**: 2.2x faster compilation (0.338s vs 0.734s)
- **Real FlashInfer kernels**: Not compatible due to dependencies

## What Works
✅ **Standalone CUDA kernels** without external dependencies  
✅ **2.2x compilation speedup** for compatible kernels  
✅ **No filesystem I/O** during compilation  
✅ **Direct PyTorch integration** via `_compile_kernel`  

## What Doesn't Work  
❌ **FlashInfer's templated kernels** (activation, quantization, norm)  
❌ **System C++ headers** (`<cstdint>`, `<cstddef>`, missing GCC macros)  
❌ **Thrust library dependencies** (not NVRTC compatible)  
❌ **FlashInfer's header hierarchy** (math.cuh, pos_enc.cuh, etc.)  

## Gaps to Fix for _compile_kernel

### 1. System Header Compatibility
**Problem**: NVRTC can't compile code using C++ standard library headers
```
/usr/include/c++/11/x86_64-redhat-linux/bits/c++config.h(284): error: identifier "__SIZE_TYPE__" is undefined
/usr/include/gnu/stubs.h(7): catastrophic error: cannot open source file "gnu/stubs-32.h"
```

**Root cause**: 
- NVRTC doesn't define GCC compiler macros (`__SIZE_TYPE__`, `__PTRDIFF_TYPE__`)
- Architecture detection fails (tries 32-bit instead of 64-bit)
- Missing proper include path setup for GNU libc headers

**Solution**: Define architecture macros and include paths:
```cpp
nvcc_options = [
    "-D__LP64__", "-D__SIZE_TYPE__=unsigned long", "-D__PTRDIFF_TYPE__=long",
    "-I/usr/include/c++/11/x86_64-redhat-linux", "-I/usr/include/gnu"
]
```

### 2. Thrust Library Integration  
**Problem**: Thrust headers require `__host__/__device__` annotations
```
error: A function without execution space annotations is considered a host function, 
and host functions are not allowed in JIT mode
```

**Solution**: Use `--default-device` flag (if supported) or avoid Thrust

### 3. Template Instantiation
**Problem**: FlashInfer uses heavy C++ templates that may not instantiate properly in NVRTC
**Solution**: Explicit template instantiation or simpler kernel designs

## Technical Details

### Implementation  
- Added `build_with_nvrtc(kernel_names)` method to `JitSpec` class
- Added `build_and_load_with_nvrtc(kernel_names)` convenience method  
- Automatic filtering of NVRTC-incompatible flags (`-O3`, `--use_fast_math`, etc.)
- Explicit kernel name specification (no auto-discovery)
- Basic include path setup for FlashInfer headers

### Current Limitations
Real FlashInfer kernels fail because:
1. **System headers**: Complex GCC-specific header dependencies
2. **Architecture detection**: 32/64-bit wordsize issues with GNU headers  
3. **Template complexity**: Heavy use of C++ templates and STL

## Recommendations

### For FlashInfer Team
1. **Don't integrate NVRTC** for existing kernels - too many dependencies
2. **Consider for new simple kernels** - avoid Thrust/complex templates
3. **Keep as experimental feature** for kernel development/prototyping

### Alternative Approaches
1. **Improve ninja build times** - parallel compilation, better caching
2. **AOT compilation** for frequently used kernel variants
3. **Incremental compilation** - only rebuild changed kernels

## Code Changes
- **Modified**: `flashinfer/jit/core.py` (lines 229-282)
- **Added**: `build_with_nvrtc()`, `build_and_load_with_nvrtc()` methods
- **Tests**: 
  - `test_nvrtc.py` - Simple kernel performance test (2.2x speedup)
  - `test_real_kernels.py` - FlashInfer kernel compatibility tests (fails)
  - `test_real_flashinfer_kernels.py` - Simplified FlashInfer-style kernels (should work)

## Conclusion
NVRTC provides significant speedup but is incompatible with FlashInfer's architecture. The current ninja-based system remains the best approach for FlashInfer's complex, templated kernels.