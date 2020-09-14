__kernel void ReLU(__global float* inputs, __global float* bias, int N)
{
    int x = get_global_id(0); 
    int y = get_global_id(1);
    int inputCh = get_global_id(2);
    int addr = N * N * inputCh + x * N +y;
    int input = inputs[addr];
    
    //bias
    input += bias[inputCh];

    //ReLU
    inputs[addr] = (input > 0) ? input : 0.0f;

    return;
}