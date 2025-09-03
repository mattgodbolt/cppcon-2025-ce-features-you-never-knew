## UI overview

```cpp
int square(int x) {
  return x * x;
}
```

<!-- .element: data-ce -->

Notes:
- new pane
  - Add new vs Tool
    - Defer talking about them til later
- dragging panes
- Selecting compilers
- **saved templates**
- settings?
- Diff view
- Compliance view

---

## Tools

```cpp
#define SQR(x) x * x
auto please_dont_use_preprocessor(double y) {
  return SQR(y + 1);
}
```
<!-- .element: data-ce -->

Notes:
- Preprocessor

---

## Tools

```cpp
// setup
  #include <array>
  #include <algorithm>
  #include <numeric>

auto silly_sum() {
  std::array<unsigned, 82> numbers;
  std::ranges::iota(numbers, 1);
  return std::ranges::fold_left_first(
    numbers, std::plus{}
  );
}
```

<!-- .element: data-ce -->

Notes:
- Stack usage
- Optimization view, -O1 -O2 -O3
- clang is better
- can show eventually clang gives up 

---

## Overrides & Flags

```cpp
// todo not this one
unsigned approx_log2(unsigned num) {
  unsigned result = 0;
  while (num) {
    result++;
    num >>= 1;
  }
  return result;
}
```

<!-- .element: data-ce data-ce-options="-pedantic -Wshadow -Wconversion" -->

Notes:
- need a better example that needs actual flags and selection of compiler doodad
- pop out flags
- show overrides

---

## Execution

```cpp
#include <cstdio>

int main() {
  puts("hello world");
}
```
<!-- .element: data-ce -->

Notes:
- show execution
- show execute only
- show libsegfault
- show ARM!

---

## GPU Execution!

```cpp
// setup
  #include <cstdio>
  #include <vector>

__global__
void saxpy(int n, float a, const float *x, float *y) {
  const auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main() {
  constexpr int N = 1<<20;
  std::vector<float> x(N, 1.f);
  std::vector<float> y(N, 2.f);
  // ... cuda magic to copy to d_x and d_y ...
  /// hide
  float *d_x, *d_y;
  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));
  cudaMemcpy(d_x, x.data(), N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y.data(), N*sizeof(float), cudaMemcpyHostToDevice);

  /// unhide
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
  /// ...
  /// hide

  cudaMemcpy(y.data(), d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  cudaFree(d_y);

  float maxError = 0.0f;
  for (std::size_t i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);
  /// unhide
}
```

<!-- .element: data-ce data-ce-language="cuda" -->

Notes:
- show GPU only (will need a link for this) TODO more inspiring demo

---

## Analysis tools

```cpp
// setup
  #include <span>
[[gnu::attribute("")]]
float sum_of_squares(std::span<const float, 1024> array) {
  float result = 0.f;
  for (const auto f : array) result += f * f;
  return result;
}
```
<!-- .element: data-ce data-ce-options="-O3" -->

Notes:
- CFG
- OPT warnings?
- IR views
- Stack use
- pipeline viewer (with useful thing)
- LLVM MCA (my favourite)

---

## Analysis tools

```cpp
unsigned population_count(unsigned input) {
    unsigned result = 0;
    while (input) {
        result++;
        input &= input - 1;
    }
    return result;
}
```
<!-- .element: data-ce data-ce-options="-O3" -->

Notes:
- CFG
- OPT warnings?
- IR views
- Stack use
- pipeline viewer (with useful thing)
- LLVM MCA (my favourite)

---

## IDE mode

Notes:
- yeah

---

## Misc trivia

- *.godbolt.org
- compiler-explorer.com, godbo.lt
