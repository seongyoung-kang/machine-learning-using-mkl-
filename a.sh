for ((times=0;times<3;times++)); do
    for ((i=80;i<=200;i+=10)); do
        for ((j=80;j<=200;j+=10)); do
            for ((k=80;k<=200;k+=10)); do
                for ((l=80;l<=200;l+=10)); do
                    for ((m=80;m<=200;m+=10)); do
                        ./mnist $i $j $k $l $m
                    done
                done
            done
        done
    done
done
        #if [ 0 -eq $i ]; then
        #    ./mnist 1
        #else
        #    ./mnist $i*10 dump_hbw_mini50 50
        #fi
