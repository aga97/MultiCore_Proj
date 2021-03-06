__kernel void convolution_layer(__global float* inputs, __global float* outputs, __global float* filters, __local float* l_sum, int N, int D1, int D2)
{
   /*
    inputs = intput image
    output = output image
    filter = filter
    l_sum = l_sum으로 넣어서 계산후 output으로
    N = N * N 크기
    D1 = input channel size
    D2 = output channel size
    local_height = 로컬 좌우
    local_width = 로컬 위아래
   */ 
    
   // 필터에서 사용하는 패딩의 양
    
   int filterRadius = 1;
   int padding = 2;

   int i = get_global_id(0);
   int j = get_global_id(1);
   int inputCh = get_global_id(2);
   int outputCh = get_global_id(3);
   
   __local int inputStartBase = N*N*inputCh;

   int groupStartRow = get_group_id(0) * get_local_size(0);
   int groupStartCol = get_group_id(1) * get_local_size(1);

   int localSizeRow=get_local_size(0);
   int localSizeCol=get_local_size(1);


   int localRow = get_local_id(0);
   int localCol = get_local_id(1);


   int globalRow = groupStartRow + localRow;
   int globalCol = groupStartCol + localCol;

    int l_sum_width;

    if(N < 32)
        l_sum_width = N + 2;
    else
        l_sum_width = 16 + 2;


   for (int i = localRow; i < l_sum_width; i += get_local_size(0)) {
 
        int curRow = groupStartRow + i;
 
        for (int j = localCol; j < l_sum_width; j += get_local_size(1)) {
 
            int curCol = groupStartCol + j;
 
            if (curRow < N && curCol < N) {
                l_sum[i*l_sum_width + j] = inputs[inputStartBase+curRow*N + curCol];
            }
        }
    }
    
   barrier(CLK_LOCAL_MEM_FENCE);
   
    

    __local l_filter[9]; 
        if(localRow==0&&localCol<9){
            l_filter[localCol] = filters[9 * D1 * outputCh + 9 * inputCh + localCol];
    }

    barrier(CLK_LOCAL_MEM_FENCE);


    //l_sum의 가장자리를 0으로 채운다.
    if(globalRow == 0)
    {
        l_sum[globalRow * l_sum_width  + localCol + 1] = 0;

        if(globalCol == 0)
            l_sum[0][0] = 0;
    }

    if(globalRow == N-1)
    {
        l_sum[(l_sum_width - 1) * l_sum_width + localCol + 1] = 0;
        if(globalCol == 0)
            l_sum[(l_sum_width - 1 ) * l_sum_width + 0] = 0;
    }

    if(globalCol == 0)  
    {
        l_sum[(localRow + 1) * l_sum_width  + globalCol] = 0;
        if(globalRow == N-1)
            l_sum[ 0 * l_sum_width + l_sum_width - 1] = 0;
    }
    if(globalCol == N-1)
    {
        l_sum[(localRow + 1) * l_sum_width  + (l_sum_width - 1)] = 0;
        if(globalRow == N-1)
            l_sum[(l_sum_width - 1) * l_sum_width + l_sum_width - 1] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if( l_sum[localCol, localRow] > 0 )
    {
        float sum = 0.0f;

        sum += l_sum[ (localRow - 1 + 1) * (localCol - 1 + 1)] * l_filter[0]
        sum += l_sum[ (localRow - 1 + 1) * (localCol + 1    )] * l_filter[1]
        sum += l_sum[ (localRow - 1 + 1) * (localCol + 1 + 1)] * l_filter[2]

        sum += l_sum[ (localRow + 1    ) * (localCol - 1 + 1)] * l_filter[3]
        sum += l_sum[ (localRow + 1    ) * (localCol + 1    )] * l_filter[4]
        sum += l_sum[ (localRow + 1    ) * (localCol + 1 + 1)] * l_filter[5]

        sum += l_sum[ (localRow + 1 + 1) * (localCol - 1 + 1)] * l_filter[6]
        sum += l_sum[ (localRow + 1 + 1) * (localCol + 1    )] * l_filter[7]
        sum += l_sum[ (localRow + 1 + 1) * (localCol + 1 + 1)] * l_filter[8]
        
     
        outputs[N*N*D1*outputCh + N*N*inputCh + globalRow * N + globalCol] = sum;
        
    }

    return;
}