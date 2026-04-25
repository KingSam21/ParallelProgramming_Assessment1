// ================================================================================================
// Generate a random number Function
// ================================================================================================
uint mlcg_rand(uint *state) {
    *state = *state * 1664525u + 101390422u;
    return *state;
}


// ================================================================================================
// Initialise Forest Kernel Function
// 0 = empty cell
// 1 = Tree
// 2 = Burning Tree.
// ================================================================================================
__kernel void initialise_forest(
                                __global float* forest, 
                                int n, 
                                const float probTree, 
                                const float probBurning)
{
    int id = get_global_id(0);

    // Safety check, so kernel remains within the bounds of the forest grid.
    if (id >= n * n) return;

    // 
    forest[id] = id%3;

    //printf("Hello from work-item %f\n", forest[id]);
    //printf("%d\n", id);
}



/*
// ================================================================================================
// Update Forest Kernel Function
// ================================================================================================
__kernel void update_forest(
                            __global float* forest, 
                            int n, 
                            const float probImmune,
                            const float probLightning)
{
    if (randomnumber < propImmune){
        forest[id] = 2
    }
}
*/