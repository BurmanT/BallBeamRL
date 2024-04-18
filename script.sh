#!/bin/zsh

# loop
for i in {0..2}; do
    echo -e "\nROUND $i\n"
    for j in {0..2}; do 
        python qlearning.py $i $j  
    done 
    wait
done 

