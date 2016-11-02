#!/bin/sh

echo test
for i in {50..120..10}
do
    for j in {80..150..10}
    do
        for k in {80..150..10}
        do
            for l in {80..150..10}
            do
                for m in {80..150..10}
                do
                    echo $i $j $k $l $m
                    ./mnist $i $j $k $l $m
                done #for m
            done #for l
        done #for k
    done #for j
done #for i

#for ((times=0;times<1;times++)); do
#    for ((i=50;i<=120;i+=10)); do
#        for ((j=80;j<=150;j+=10)); do
#            for ((k=80;k<=150;k+=10)); do
#                for ((l=80;l<=150;l+=10)); do
#                    for ((m=80;m<=150;m+=10)); do
#                        ./mnist $i $j $k $l $m;
#                        #echo $i $j $k $l $m
#                    done
#                done
#            done
#        done
#    done
#done
        #if [ 0 -eq $i ]; then
        #    ./mnist 1
        #else
        #    ./mnist $i*10 dump_hbw_mini80 80
        #fi
#for ((m=80;m<=150;m+=10)); do
    #./mnist $i $j $k $l $m;
    ./mnist 100 100 100 100 100;
#echo $i $j $k $l $m
#done
