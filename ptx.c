/*
t gcc-8 ptx.c -o ptx -lcuda -l:libnvptxcompiler_static.a -lm -lpthread  &&  t ./ptx
ptxas k00.ptx
stackoverflow.com/questions/20012318
t nvcc k00.ptx -arch=sm_70 -cubin -o k00.cubin
*/
#include <stdio.h>
#include <string.h>
#include "cuda.h"
#include "nvPTXCompiler.h"

#define NTHRS 0x40
#define NBLKS 0x02

#define dchk(x)  do{                                      \
	CUresult st = x;                                        \
	if(st!=CUDA_SUCCESS){                                   \
		const char* msg;                                      \
		cuGetErrorName(st,&msg);                              \
		printf("error: %s failed with error %s\n", #x, msg);  \
		exit(1);                                              \
	}                                                       \
}while(0)

#define ptxchk(x)  do{  \
	nvPTXCompileResult st = x;                                  \
	if(st!=NVPTXCOMPILE_SUCCESS){                               \
		printf("error: %s failed with error code %d\n", #x, st);  \
		exit(1);                                                  \
	}                                                           \
}while(0)

// ----------------------------------------------------------------------------------------------------------------------------# @blk1
// extern "C" __global__ void fn00(float* x, float* y, float* out){
//   size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//   out[tid] = x[tid] + y[tid];
// }
const char* ptxCode = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
.visible .entry fn00(\n\
	.param .u64 fn00__param_0,\n\
	.param .u64 fn00__param_1,\n\
	.param .u64 fn00__param_2\n\
){\n\
	.reg .f32          %f<4>;\n\
	.reg .b32          %r<5>;\n\
	.reg .b64          %rd<11>;\n\
	ld.param.u64       %rd1, [fn00__param_0];\n\
	ld.param.u64       %rd2, [fn00__param_1];\n\
	ld.param.u64       %rd3, [fn00__param_2];\n\
	cvta.to.global.u64 %rd4, %rd3;\n\
	cvta.to.global.u64 %rd5, %rd2;\n\
	cvta.to.global.u64 %rd6, %rd1;\n\
\n\
	mov.u32            %r1, %ctaid.x;\n\
	mov.u32            %r2, %ntid.x;\n\
	mov.u32            %r3, %tid.x;\n\
\n\
	mad.lo.s32         %r4,   %r2,    %r1,  %r3;\n\
	mul.wide.u32       %rd7,  %r4,    4;\n\
	add.s64            %rd8,  %rd6,   %rd7;\n\
\n\
	ld.global.f32      %f1,   [%rd8];\n\
	add.s64            %rd9,  %rd5,   %rd7;\n\
\n\
	ld.global.f32      %f2,   [%rd9];\n\
	add.f32            %f3,   %f1,    %f2;\n\
	add.s64            %rd10, %rd4,   %rd7;\n\
\n\
	st.global.f32      [%rd10], %f3;\n\
	ret;\n\
}\n";
	
// ----------------------------------------------------------------------------------------------------------------------------# @blk1
void ker_cubin_launch(char* name, void* elf_data, size_t A_idim, int nblks,int nthrs){
	dchk(cuInit(0));
	CUdevice   dev; dchk(cuDeviceGet(&dev, 0));
	CUcontext  ctx; dchk(cuCtxCreate(&ctx, 0, dev));
	CUmodule   mod; dchk(cuModuleLoadDataEx(&mod, elf_data, 0, 0, 0));
	CUfunction ker; dchk(cuModuleGetFunction(&ker, mod, name));

	// ----------------------------------------------------------------
	size_t A_bdim = A_idim*sizeof(float);  // generate input for execution, and create output buffers
	float* A_cpu  = malloc(A_bdim);
	float* B_cpu  = malloc(A_bdim);
	float* C_cpu  = malloc(A_bdim);
	for(int i=0; i<A_idim; ++i){
		A_cpu[i] = 1*i;
		B_cpu[i] = 2*i;
	}
	CUdeviceptr A_gpu; dchk(cuMemAlloc(&A_gpu, A_bdim));
	CUdeviceptr B_gpu; dchk(cuMemAlloc(&B_gpu, A_bdim));
	CUdeviceptr C_gpu; dchk(cuMemAlloc(&C_gpu, A_bdim));
	dchk(cuMemcpyHtoD(A_gpu, A_cpu, A_bdim));
	dchk(cuMemcpyHtoD(B_gpu, B_cpu, A_bdim));

	dchk(cuLaunchKernel(ker, nblks,1,1, nthrs,1,1, 0,NULL, (void*[]){&A_gpu,&B_gpu,&C_gpu},0));
	dchk(cuCtxSynchronize());

	dchk(cuMemcpyDtoH(C_cpu, C_gpu, A_bdim));  // Retrieve and print output
	for(int i=0; i<A_idim; ++i)  printf("out \x1b[31m%3ld \x1b[32m%4.0f\x1b[0m\n", i, C_cpu[i]);

	free(A_cpu);
	free(B_cpu);
	free(C_cpu);
	dchk(cuMemFree(A_gpu));
	dchk(cuMemFree(B_gpu));
	dchk(cuMemFree(C_gpu));

	// ----------------------------------------------------------------
	dchk(cuModuleUnload(mod));
	dchk(cuCtxDestroy(ctx));
}
	
// ----------------------------------------------------------------------------------------------------------------------------# @blk1
#define CGPU_NOPT      2
const char* CGPU_OPT[] = {"--gpu-name=sm_70", "--verbose"};

int main(int nargs, char* args[]){

	// ----------------------------------------------------------------
	nvPTXCompilerHandle compiler;
	unsigned int minorVer, majorVer;
	ptxchk(nvPTXCompilerGetVersion(&majorVer, &minorVer));  printf("PTX Compiler API Version: %d.%d\n", majorVer, minorVer);
	ptxchk(nvPTXCompilerCreate(&compiler, strlen(ptxCode),ptxCode));

	nvPTXCompileResult st=nvPTXCompilerCompile(compiler, CGPU_NOPT,CGPU_OPT);  /* numCompileOptions,compileOptions */
	if(st!=NVPTXCOMPILE_SUCCESS){
		size_t errorSize;
		ptxchk(nvPTXCompilerGetErrorLogSize(compiler, &errorSize));
		if(errorSize!=0){
			char* errorLog = malloc(errorSize+1);
			ptxchk(nvPTXCompilerGetErrorLog(compiler, errorLog));
			printf("\x1b[31m%s\x1b[0m\n", errorLog);
			free(errorLog);
		}
		exit(1);
	}

	size_t elf_bdim; ptxchk(nvPTXCompilerGetCompiledProgramSize(compiler, &elf_bdim));
	char*  elf_data = malloc(elf_bdim);
	ptxchk(nvPTXCompilerGetCompiledProgram(compiler, (void*)elf_data));

	size_t infoSize; ptxchk(nvPTXCompilerGetInfoLogSize(compiler, &infoSize));
	if(infoSize!=0){
		char* infoLog = malloc(infoSize+1);
		ptxchk(nvPTXCompilerGetInfoLog(compiler, infoLog));
		printf("\x1b[34m%s\x1b[0m\n", infoLog);
		free(infoLog);
	}

	ptxchk(nvPTXCompilerDestroy(&compiler));

	// ----------------------------------------------------------------
	ker_cubin_launch("fn00", elf_data, NBLKS*NTHRS, NBLKS,NTHRS);  // Load the compiled GPU assembly code 'elf_data'
	free(elf_data);
	exit(0);
}
