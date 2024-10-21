__kernel void vecAdd(__global const float* a, 
    __global const float* b, 
    __global float* c, 
    const unsigned int n)
{
    //Get our global thread ID                                
    int id = get_global_id(0);

    //Make sure we do not go out of bounds                    
    if (id < n)
    {
        c[id] = a[id] + b[id];
    }
}
