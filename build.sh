#! /bin/bash

if [ ! -d "test" ]; then
    mkdir "test"
fi
cmake -B test -S .
cmake --build test
if ! "test/gpu"; then
    echo "Erreur."
else
    echo "Correct."
fi

rm -rf test