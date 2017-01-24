double * temp_error;

for (i = net->num_layer-2; i > 0; i--)
{
		for (j = 0; j < net->mini_batch_size; j++)
        {
            temp_error = (double*)malloc(sizeof(double)*net->layer_size[i]);

            void cblas_dgemv (CblasRowMajor, CblasNoTrans,  net->layer_size[i], net->layer_size[i+1], 1.0,(const double *)&WEIGHT(net, i, 0, 0), net->layer_size[i+1],(const double *)&ERROR(net,i+1, j, 0,) , const MKL_INT incx , const double beta , double *y , const MKL_INT incy );

        }

}
