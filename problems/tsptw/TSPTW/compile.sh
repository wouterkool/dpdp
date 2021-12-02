set -x
rm -rf bin
mkdir bin
cd bin
g++ -O3 -std=c++0x -Wno-unused-result -Wno-write-strings -Wno-deprecated -c -lm ../*.cpp
g++ -O3 -std=c++0x -o Run *.o
cd ../
set +x