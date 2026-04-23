// 0 = empty cell
// 1 = Tree
// 2 = Burning Tree.

// ================================================================================================
// Initialise Forest Kernel Function
// ================================================================================================
__kernel void initialise_forest(
                                __global float* forest, 
                                int n, 
                                const float probTree, 
                                const float probBurning)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Safety check, so kernel remains within the bounds of the grid.
    if (x >= n || y >= n) return;

    // 
    int id = x + y * n;
    forest[id] = 2%3;

    printf("Hello from work-item %d\n", x);
}

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