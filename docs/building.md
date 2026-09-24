# Building the external solvers

DRISPI calls three binaries. PyVRP is a Python dependency and needs no extra build. The others live in git submodules under `ext/` and are not committed as binaries.

| Solver | What DRISPI runs | Override |
|--------|------------------|----------|
| FILO | `ext/filo/build/filo` | `FILO_BIN` |
| FILO2 | `ext/filo2/build/filo2` | `FILO2_BIN` |
| AILS-II | `ext/ails2/build/AILSII.jar` | `AILS2_JAR` |

Clone the submodules first:

```bash
git submodule update --init --recursive
```

Host packages: a C++17 compiler, CMake ≥ 3.22, and a JDK 17 (the AILS-II build uses `javac --release 17`). FILO also needs the COBRA library installed on the machine, because its CMake file does `find_package(cobra)`.

## COBRA

FILO links against COBRA. FILO2 does not.

```bash
cd ext/cobra
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
sudo cmake --install build
```

## FILO

The Python wrapper passes `--time`, which exists only when FILO is built with time-based termination.

```bash
cd ext/filo
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTIMEBASED_TERMINATION=ON
cmake --build build -j
```

Leave `ENABLE_GUI` off. The binary is `ext/filo/build/filo`.

## FILO2

The wrapper passes `--optimization-seconds`, which exists only with `ENABLE_TIMELIMIT`.

```bash
cd ext/filo2
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_TIMELIMIT=ON
cmake --build build -j
```

The binary is `ext/filo2/build/filo2`.

## AILS-II

```bash
cd ext/ails2
./build.sh
```

That compiles the sources listed in `ext/ails2/sources.txt` and writes `ext/ails2/build/AILSII.jar`. The same classes are also packed as `AILSII-touch.jar`; the pipeline uses `AILSII.jar`.

Build directories are gitignored. A reviewer has to compile these after cloning.
