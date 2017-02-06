//feedforward(net,threads[0],modew0]); //이런식으로 되어야함.

void settingmodes(struct network *net,int *threads,int maxthreads,int *mode)
{
    int i;
    double feedforward_t = -1;
    double back_pass_t1 = -1;
    double back_pass_t2 = -1;
    double backpropagation_t1 = -1;
    double backpropagation_t2 = -1;

    double temp = 0;

    //threads settting
    for(i=1;i<maxthreads;i+=10)
    {
        temp = calculate_feedforward(net,i,1);
        if(feedforward_t == -1 || feedforward_t > temp)
        {
           feedforward_t = temp;
           threads[0] = i;
        }

        temp = calculate_back_pass_1(net,i,1);
        if(back_pass_t1 == -1 || back_pass_t1 > temp)
        {
           back_pass_t1 = temp;
           threads[1] = i;
        }

        temp = calculate_back_pass_2(net,i,1);
        if(back_pass_t2 == -1 || back_pass_t2 > temp)
        {
           back_pass_t2 = temp;
           threads[2] = i;
        }

        temp = calculate_backpropagation_1(net,i,1);
        if(backpropagation_t1 == -1 || backpropagation_t1 > temp)
        {
           backpropagation_t1 = temp;
           threads[3] = i;
        }

        temp = calculate_backpropagation_2(net,i,1);
        if(backpropagation_t2 == -1 || backpropagation_t2 > temp)
        {
           backpropagation_t2 = temp;
           threads[4] = i;
        }
    }

    //mode select mode값들은 1으로 초기화되어 있습니다.

    feedforward(net,threads[0],0);
    temp = &feedforward->diff;
    feedforward(net,threads[0],1);
    if(temp < &feedforward->diff)
        mode[0]=0;

    back_pass(net,threads[1],threads[2],0);
    temp = &back_pass->diff;
    feedforward(net,threads[1],threads[2],1);
    if(temp < &back_pass->diff)
        mode[1]=0;

    backpropagation(net,threads[3],threads[4],0);
    temp = &backpropagation->diff;
    feedforward(net,threads[3],threads[4],1);
    if(temp < &backpropagation->diff)
        mode[2]=0;
}

double calculate_feedforward(struct network *net, int thread,int mode)
{
    START_TIME(feedforward);
    feedforward(net, thread,1);
    END_TIME(feedforward);
    return &feedforward->diff;
}
double calculate_back_pass_1(struct network *net, int thread,int mode)
{
    START_TIME(back_pass);
    back_pass(net, thread,1,1);
    END_TIME(back_pass);
    return &back_pass->diff;
}

double calculate_back_pass_2(struct network *net, int thread,int mode)
{
    START_TIME(back_pass);
    back_pass(net, 1,thread,1);
    END_TIME(back_pass);
    return &back_pass->diff;
}

double calculate_backpropagation_1(struct network *net, int thread,int mode)
{
    START_TIME(backpropagation);
    feedforward(net, thread,1,mode);
    END_TIME(backpropagation);
    return &backpropagation->diff;
}

double calculate_backpropagation_2(struct network *net, int thread,int mode)
{
    START_TIME(backpropagation);
    feedforward(net,1,thread,mode);
    END_TIME(backpropagation);
    return &backpropagation->diff;
}
