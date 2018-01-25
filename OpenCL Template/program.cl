#define GLINTEROP

// function that returns the value of the bit at the provided x and y coordinates in the provided buffer
uint GetBit ( __global uint* second, uint x, uint y, uint levelWidth)
{
	// recreate the id using the provided x and y positions
	uint id = y * levelWidth + x;
	// get the index of the correct uint
	uint uintIndex = id >> 5;

	// return the value of the bit
	return (second[uintIndex] >> (int)(x & 31)) & 1U;
}

// main kernel
#ifdef GLINTEROP
__kernel void device_function(write_only image2d_t a, uint levelWidth, uint ph, uint amountOfCells, __global uint* pattern, __global uint* second, uint xoffset, uint yoffset, int screenWidth, int screenHeight)
#else
__kernel void device_function(__global int* a, uint levelWidth, uint ph, uint amountOfCells, __global uint* pattern, __global uint* second, uint xoffset, uint yoffset, int screenWidth, int screenHeight)
#endif
{
	// get the id of the current cell
	int id = get_global_id(0);

	// check if it is outside our bounds, if so, return
    if (id >= amountOfCells)
	{
		return;
	}

	// pw is the levelwidth in uints
	// get the position on a row
	uint x = id % levelWidth;
	// get the correct row
	uint y = id / levelWidth;
	// these x and y are the position of the current cell

	// get the index of the correct uint
	uint uintIndex = id >> 5;

	// set the value of the current bit to 0 in the pattern
	atomic_and(&pattern[uintIndex], ~(1U << (x & 31)));

	// skip the outer rows and first and last pixel of each row
	if (x > 1 && x < levelWidth - 1 && y > 1 && y < ph - 1)
	{
		// get the neighbours using the GetBit function. This function gets the bits from the second buffer
		int n = GetBit(second, x - 1, y - 1, levelWidth) + GetBit(second, x - 1, y, levelWidth) + GetBit(second, x - 1, y + 1, levelWidth) + GetBit(second, x, y - 1, levelWidth) + GetBit(second, x, y + 1, levelWidth) + GetBit(second, x + 1, y - 1, levelWidth) + GetBit(second, x + 1, y, levelWidth) + GetBit(second, x + 1, y + 1, levelWidth);

		// set the current bit to 1 if the amount of neighbours allows it
		if ((GetBit(second, x, y, levelWidth) == 1 && n == 2) || n == 3)
		{
			atomic_or(&pattern[uintIndex], 1U << (int)(x & 31));
		}
	}
	
	// if we don't want to draw the pixel because it is outide our bounds, return
	if (x < xoffset || x > xoffset + screenWidth - 1 || y < yoffset || y > yoffset + screenHeight - 1)
	{
		return;
	}

#ifdef GLINTEROP
	// create the position we want to draw the pixel using the offsets
	int2 pos = (int2)((int)x - xoffset, (int)y - yoffset);
	// get the colour using the GetBit function, this time it gets it's info from the pattern buffer
	float colour = (float)GetBit(pattern, x, y, levelWidth);
	float4 col = (float4)(colour, colour, colour, 1.0f);
	// finally, write the pixel to the image
	write_imagef(a, pos, col);
#else
	// create an int, either 0 or 255, for the colour
	int c = (int)clamp(255.0f * (int)GetBit(pattern, x, y, levelWidth), 0.0f, 255.0f);
	// create the positions
	int xPos = (int)x - xoffset;
	int yPos = (int)y - yoffset;
	// set the colour to the position in the buffer
	a[xPos + yPos * screenWidth] = (c << 16) + (c << 8) + c;
#endif
}

// function that copies the data from the pattern buffer to the second buffer
__kernel void copy_data(uint levelWidth, uint amountOfCells, __global uint* pattern, __global uint* second)
{
	// get the id of the current cell
	int id = get_global_id(0);

	// check if it is outside our bounds, if so, return
    if (id >= amountOfCells)
	{
		return;
	}

	// get the position on a row
	uint x = id % levelWidth;
	// get the correct row
	uint y = id / levelWidth;
	// these x and y are the position of the current cell

	// get the index of the correct uint
	uint uintIndex = id >> 5;

	// get the value from the current bit from the pattern buffer
	int bitValue = GetBit(pattern, x, y, levelWidth);

	// if the cell is alive, set its bit in second to 1
	if (bitValue == 1)
	{
		atomic_or(&second[uintIndex], 1U << (int)(x & 31));
	}
	// if it is dead, set its bit in second to 0
	else
	{
		atomic_and(&second[uintIndex], ~(1U << (x & 31)));
	}
}