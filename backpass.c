
double * temp3;
double * temp_error;

for (i = net->num_layer-2; i > 0; i--)
    {
		for (j = 0; j < net->mini_batch_size; j++)
        {
            temp3 = (double*)malloc(sizeof(double)*net->layer_size[i]);
            temp_error = (double*)malloc(sizeof(double)*net->layer_size[i]*net->layer_size[i+1]);

            for(k=0;k<net->layer_size[i];k++)
                temp3[k] = 1.0;

         cblas_dger(CblasRowMajor,net->layer_size[i], net->layer_size[i+1],
         1.0, tmp3, 1, (const double *)&ERROR(net,i, j, 0,),1,temp_error,net->layer_size[i+1])



cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, net->layer_size[i], net->layer_size[i], net->layer_size[i+1], 1.0, temp_error,net->layer_size[i+1], (const double *)&WEIGHT(net, i, 0, 0), net->layer_size[i], 0.0,&ERROR(net, i, j, 0), net->layer_size[i]); //error = a*w


        for(k=0;k<net->layer_size[i]*net->layer_size[i];k++)
        {
            if(k/net->layer_size[i] !=0)
            ERROR(net,i,j,k%net->layer_size[i]) += ERROR(net,i,j,k)
        }



		}
}
