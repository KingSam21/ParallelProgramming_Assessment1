//=================================================================================================
// Generate a random number Function
//=================================================================================================
uint mlcg_rand(uint *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

//=================================================================================================
// Turn the random number into a float between 0 and 1.
//=================================================================================================
float random_float(uint *state) {
    return (float)mlcg_rand(state)/4294967296.0f;
}


//=================================================================================================
// Initialise Forest Kernel Function
// 0 = empty cell
// 1 = Tree
// 2 = Burning Tree.
//=================================================================================================
__kernel void initialise_forest(
                                __global int* forest,                                               // Sets the current forest grid.
                                int n,                                                              // Sets the size of a row/column in the forest grid.
                                const float probTree,                                               // Sets the value for the probability of spawning a tree.
                                const float probBurning)                                            // Sets the probability of the tree burning.
{
    //---------------------------------------------------------------------------------------------
    // Retreive the id of the current workitem.
    //---------------------------------------------------------------------------------------------
    int id = get_global_id(0);

    //---------------------------------------------------------------------------------------------
    // Safety check, so kernel remains within the bounds of the forest grid.
    //---------------------------------------------------------------------------------------------
    if (id >= n * n) return;

    //---------------------------------------------------------------------------------------------
    // Create a seed, to send for randomising (utilising the id, so it is different per cell)
    //---------------------------------------------------------------------------------------------
    uint seed = id + 21295u;

    //---------------------------------------------------------------------------------------------
    // Generate random numbers to determine the what the state of cell is.
    //---------------------------------------------------------------------------------------------
    if (random_float(&seed) < probTree){                                                            // Check if the random number generated is lower than Probtree.
        if (random_float(&seed) < probBurning) {                                                    // Then, check if a new random number is lower than Prob Burning. 
            forest[id] = 2;                                                                         // If it is lower set the forest[id] to equal a burning tree.
        }
        else {
            forest[id] = 1;                                                                         // Otherwise set the tree, to be a normal one.
        }
    }
    else {                                                                                          // If the first random number is higher than probTree.
        forest[id] = 0;                                                                             // The cell is initialised as empty.              
    }
}



//=================================================================================================
// Update Forest Kernel Function
// 0 = empty cell
// 1 = Tree
// 2 = Burning Tree.
//=================================================================================================
__kernel void update_forest(
                            __global int* currentForest,                                            // The current forest grid.
                            __global int* nextForest,                                               // The Updated forest grid.
                            int n,                                                                  // The size of the forest grid.
                            const float probImmune,                                                 // Sets the value for the probability of a tree being immune.
                            const float probLightning)                                              // Sets the probability of the tree getting struck by lightning.
{
    //---------------------------------------------------------------------------------------------
    // Retreive the id of the current workitem.
    //---------------------------------------------------------------------------------------------
    int id = get_global_id(0);

    //---------------------------------------------------------------------------------------------
    // Safety check, so kernel remains within the bounds of the forest grid.
    //---------------------------------------------------------------------------------------------
    if (id >= n * n) return;

    //---------------------------------------------------------------------------------------------
    // Create Variables.
    //---------------------------------------------------------------------------------------------
    int x = id % n;                                                                                 // Obtain the x position of the cell (Needed to check neighbouring cells).
    int y = id / n;                                                                                 // Obtain the y position of the cell (Needed to check neighbouring cells).
    int currentCell = currentForest[id];                                                            // Obtain the current state of the cell.
    uint seed = id + 215195u;                                                                       // Create a seed, to send for randomising.

    //---------------------------------------------------------------------------------------------
    // Perform checks, to determine the state for the next iteration of the cell.
    //---------------------------------------------------------------------------------------------
    if (currentCell == 0 || currentCell == 2) {                                                     // Check to see if the current cell is burning or empty.
        nextForest[id] = 0;                                                                         // Set the updated cell to be empty.
    }
    else {                                                                                          // If the current cell contains a tree, perform checks.
        if (random_float(&seed) < probImmune) {                                                     // Check to see if the tree is immune for this iteration.
            nextForest[id] = 1;                                                                     // if it is immune, set the next cell iteration to also be a tree.
        }
        else {                                                                                      // If the tree is not immune, perform further checks.
            bool neighbourOnFire = false;                                                           // Variable for if neighbouring trees are on fire.                                                             
            //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            // loop through neighbouring cells.
            //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for (int j = -1; j <= 1; j++) {                                                         // Loop through all j (y axis) cells.
                for (int i = -1; i <= 1; i++) {                                                     // loop through all i (x axis) cells.
                    if (j == 0 && i == 0) {                                                         // Check to see if the current cell is selected.
                        continue;                                                                   // Skip to the next iteration.
                    }
                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    // Apply formula for periodic boundaries
                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    int neighbouringCell_x = (x + i + n) % n;                                       // Calculate the new position of the x cell, accounting for periodic boundaries.                     
                    int neighbouringCell_y = (y + j + n) % n;                                       // Calculate the new position of the y cell, accounting for periodic boundaries.

                    if (currentForest[neighbouringCell_y * n + neighbouringCell_x] == 2) {          // Check to see if the neighbouring cell in the current forest is burning.
                        neighbourOnFire = true;                                                     // If the neighbouring cell is burning, set the variable to equal true. 
                    }
                }
            }
            if (neighbourOnFire || random_float(&seed) < probLightning) {                           // Perform final check with randomised number to see if lightning happened or a neighbouring cell was on fire.
                nextForest[id] = 2;                                                                 // If true, set current cell, to be burning.
            }
            else {                                                                                  // If neighbouring cells aren't on fire and lighning didn't happen.
                nextForest[id] = 1;                                                                 // Set the current cell to be a standard tree.
            }
        }
    }

}


//=================================================================================================
// To loop around to the beginning/end of a list (PERIODIC BOUNDARY), the formula is:
// (current cell + offset + total size of list) % total size of list.
//
// e.g.:
// [0], [1], [2].
// When starting from index [0] in the list, and checking to the left. (it returns the value on the far side.)
// 0 - 1 + 3 = 2.
// 2 % 3 = 2.
//
// When starting from index [2] in the list and checking to to the right. (it returns an index of [0]).
// 2 + 1 + 3 = 6.
// 6 % 3 = 0.
//=================================================================================================