void back_pass(struct network *net, int thread1, int thread2)
{
	int i, j, k, l;
	double sum = 0.0;
    timeutils *back_pass = &net->t_back_pass;

	START_TIME(back_pass);

    double * temp1;//neuron - error
    double * temp2;//sigmoid zs
    double * temp_error;

    temp1 = (double*)malloc(sizeof(double) * net->mini_batch_size * net->layer_size[net->num_layer-1]);
    temp2 = (double*)malloc(sizeof(double) * net->mini_batch_size * net->layer_size[net->num_layer-1]);

    // neuron - error
    vdSub(net->layer_size[net->num_layer-1]*net->mini_batch_size,&NEURON(net, net->num_layer-1, 0, 0),&ERROR(net, net->num_layer-1, 0, 0),temp1);

    //sigmoid zs
    #pragma omp parallel for num_threads(thread1)
	    for (i = 0; i < net->mini_batch_size*net->layer_size[net->num_layer-1]; i++)
            {
		         temp2[i]=sigmoid_prime(ZS(net, net->num_layer-1, 0, i));
	    	}

    //temp1 * temp2 (when this loop is end  first delta is done!!)
    vdMul(net->layer_size[net->num_layer-1]*net->mini_batch_size,temp1,temp2,&ERROR(net, net->num_layer-1, 0, 0))

    //caculrate delta to using backpropagation algorithm
    for (i = net->num_layer-2; i > 0; i--)
    {
		for (j = 0; j < net->mini_batch_size; j++)
        {
            //temp_error = weight * past_error
            temp_error = (double*)malloc(sizeof(double)*net->layer_size[i]);

            //calculate temp_error
            cblas_dgemv (CblasRowMajor, CblasNoTrans,  net->layer_size[i], net->layer_size[i+1], 1.0,(const double *)&WEIGHT(net, i, 0, 0), net->layer_size[i+1],(const double *)&ERROR(net,i+1, j, 0,) ,1 ,0.0 , temp_error , 1);

            //calculate delta = past error * weight * sigmoidprime(zs)
            #pragma omp parallel for num_threads(thread1)
            for(k=0;k<net->layer_size[i];k++)
            {
                ERROR(net, i, j, k) = temp_error[k]*sigmoid_prime(ZS(net, i, j, k));
            }

        }
    }
}
