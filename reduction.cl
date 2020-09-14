__kernel void reduction_layer(__global float* inputs, __global float* outputs, __local float* l_sum, int N, int D1, int D2, int outputCh)
{
    int x = get_global_id(0); 
    int y = get_global_id(1);
    int inputCh = get_global_id(2);
 
    int l_i = get_local_id(2);

    if( D1 != 3)
    {
        l_sum[2*inputCh]=inputs[N*N*D1*outputCh + N*N*(2*inputCh) + N*x + y];
        l_sum[2*inputCh+1]=inputs[N*N*D1*outputCh + N*N*(2*inputCh+1) + N*x + y];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int p = get_local_size(2)*2 ; p >= 1; p >> 1)
        { 
            if (l_i < p)
            {
                l_sum[l_i] += l_sum[l_i+p];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
	    }

        outputs[N*N*outputCh + N * x + y] = l_sum[0];
    }


    for (int i = 0; i < get_global_size(2); i++)
		{
			l_sum[0] += l_sum[inputCh*2];
			l_sum[1] += l_sum[inputCh*2 +1];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		
		l_sum[0] += l_sum[1];

    else
    {
        l_sum[inputCh] = inputs[N * N * D1 * outputCh + N * N * inputCh + N*x + y];
        barrier(CLK_LOCAL_MEM_FENCE);

        
        if(l_i==0) l_sum[0] += l_sum[1] + l_sum[2];

        outputs[N*N*outputCh + N*x + y] = l_sum[0];

    }    


    return;
}