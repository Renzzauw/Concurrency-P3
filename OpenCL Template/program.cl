#define GLINTEROP

// function that returns the value of the bit at the provided x and y coordinates in the provided buffer
uint GetBit ( __global uint* second, uint x, uint y, uint pw)
{
	// recreate the id using the provided x and y positions
	uint id = y * (pw * 32) + x;
	// get the index of the correct uint
	uint uintIndex = id >> 5;

	// return the value of the bit
	return (second[uintIndex] >> (int)(x & 31)) & 1U;
}

// main kernel
__kernel void device_function( write_only image2d_t a, uint pw, uint ph, uint amountOfCells, __global uint* pattern, __global uint* second, uint xoffset, uint yoffset, int screenWidth, int screenHeight )
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
	uint x = id % (pw * 32);
	// get the correct row
	uint y = id / (pw * 32);
	// these x and y are the position of the current cell

	// get the index of the correct uint
	uint uintIndex = id >> 5;

	// set the value of the current bit to 0 in the pattern
	atomic_and(&pattern[uintIndex], ~(1U << (x & 31)));
	//pattern[uintIndex] &= ~(1U << (x & 31));

	// skip the outer rows and first and last pixel of each row
	if (x > 1 && x < pw * 32 - 1 && y > 1 && y < ph - 1)
	{
		// get the neighbours using the GetBit function. This function gets the bits from the second buffer
		int n = GetBit(second, x - 1, y - 1, pw) + GetBit(second, x - 1, y, pw) + GetBit(second, x - 1, y + 1, pw) + GetBit(second, x, y - 1, pw) + GetBit(second, x, y + 1, pw) + GetBit(second, x + 1, y - 1, pw) + GetBit(second, x + 1, y, pw) + GetBit(second, x + 1, y + 1, pw);

		// set the current bit to 1 if the amount of neighbours allows it
		if ((GetBit(second, x, y, pw) == 1 && n == 2) || n == 3)
		{
			atomic_or(&pattern[uintIndex], 1U << (int)(x & 31));
			//pattern[uintIndex] |= 1U << (int)(x & 31);
		}
	}
	
	// if we don't want to draw the pixel because it is outside our bounds, return
	if (x < xoffset || x > (xoffset + screenWidth - 1) || y < yoffset || y > (yoffset + screenHeight - 1))
	{
		return;
	}

	// create the position we want to draw the pixel using the offsets
	int2 pos = (int2)((int)x - xoffset, (int)y - yoffset);
	// get the colour using the GetBit function, this time it gets it's info from the pattern buffer
	float colour = (float)GetBit(pattern, x, y, pw);
	float4 col = (float4)(colour, colour, colour, 1.0f);
	// finally, write the pixel to the image
	write_imagef(a, pos, col);
}

// function that copies the data from the pattern buffer to the second buffer
__kernel void copy_data(int pw, int amountOfCells, __global uint* pattern, __global uint* second)
{
	// get the id of the current cell
	int id = get_global_id(0);

	// check if it is outside our bounds, if so, return
    if (id >= amountOfCells)
	{
		return;
	}

	// get the position on a row
	uint x = id % (pw * 32);
	// get the correct row
	uint y = id / (pw * 32);
	// these x and y are the position of the current cell

	// get the index of the correct uint
	uint uintIndex = id >> 5;

	// get the value from the current bit from the pattern buffer and set it to the second buffer
	int bitValue = GetBit(pattern, x, y, pw);
	atomic_or(&second[uintIndex], bitValue << (int)(x & 31));
	//second[uintIndex] |= bitValue << (int)(x & 31);
}