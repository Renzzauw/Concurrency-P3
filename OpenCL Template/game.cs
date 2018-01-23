﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using Cloo;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;

namespace Template
{
    class Game
    {
        // parameters for the screenwidth and -height
        public static int screenWidth;
        public static int screenHeight;
        // Zoomfactor for the camera
        public float zoom = 1f;
        // when GLInterop is set to true, the fractal is rendered directly to an OpenGL texture
        bool GLInterop = true;
        // load the OpenCL program; this creates the OpenCL context
        static OpenCLProgram ocl = new OpenCLProgram( "../../program.cl" );
        // find the kernel named 'device_function' in the program
        OpenCLKernel kernel = new OpenCLKernel( ocl, "device_function" );
        // find the kernel named 'copy_data' in the program
        OpenCLKernel copyKernel = new OpenCLKernel(ocl, "copy_data");
        // create a regular buffer; by default this resides on both the host and the device
        OpenCLBuffer<int> buffer = new OpenCLBuffer<int>( ocl, screenWidth * screenHeight );
        OpenCLBuffer<uint> secondBuffer;
        OpenCLBuffer<uint> patternBuffer;
        // create an OpenGL texture to which OpenCL can send data
        OpenCLImage<int> image = new OpenCLImage<int>( ocl, screenWidth, screenHeight );
        public Surface screen;
        Stopwatch timer = new Stopwatch();
        int generation = 0;
        // two buffers for the pattern: simulate reads 'second', writes to 'pattern'
        uint[] pattern;
        uint[] second;
        uint pw, ph; // note: pw is in uints; width in bits is 32 this value.
        uint pwph;
        uint amountOfCells;
        uint levelWidth; // in cells
        // helper function for setting one bit in the pattern buffer
        void BitSet(uint x, uint y, uint[] array) { array[y * pw + (x >> 5)] |= 1U << (int)(x & 31); }
        // mouse handling: dragging functionality
        uint xoffset = 0, yoffset = 0;
        bool lastLButtonState = false;
        int dragXStart, dragYStart, offsetXStart, offsetYStart;

        public void SetMouseState(int x, int y, bool pressed)
        {
            if (pressed)
            {
                if (lastLButtonState)
                {
                    int deltax = x - dragXStart, deltay = y - dragYStart;
                    xoffset = (uint)Math.Min(levelWidth - screenWidth, Math.Max(0, offsetXStart - deltax));
                    yoffset = (uint)Math.Min(ph - screenHeight, Math.Max(0, offsetYStart - deltay));
                }
                else
                {
                    dragXStart = x;
                    dragYStart = y;
                    offsetXStart = (int)xoffset;
                    offsetYStart = (int)yoffset;
                    lastLButtonState = true;
                }
            }
            else lastLButtonState = false;
        }
        // minimalistic .rle file reader for Golly files (see http://golly.sourceforge.net)
        public void Init()
        {
            StreamReader sr = new StreamReader("../../data/turing_js_r.rle");
            uint state = 0, n = 0, x = 0, y = 0;
            while (true)
            {
                String line = sr.ReadLine();
                if (line == null) break; // end of file
                int pos = 0;
                if (line[pos] == '#') continue; /* comment line */
                else if (line[pos] == 'x') // header
                {
                    String[] sub = line.Split(new char[] { '=', ',' }, StringSplitOptions.RemoveEmptyEntries);
                    pw = (UInt32.Parse(sub[1]) + 31) >> 5;
                    ph = UInt32.Parse(sub[3]);
                    pwph = pw * ph;
                    pattern = new uint[pwph];
                    second = new uint[pwph];
                }
                else while (pos < line.Length)
                    {
                        Char c = line[pos++];
                        if (state == 0) if (c < '0' || c > '9') { state = 1; n = Math.Max(n, 1); } else n = (uint)(n * 10 + (c - '0'));
                        if (state == 1) // expect other character
                        {
                            if (c == '$') { y += n; x = 0; } // newline
                            else if (c == 'o') for (int i = 0; i < n; i++) BitSet(x++, y, second); else if (c == 'b') x += n;
                            state = n = 0;
                        }
                    }
            }
            //pwph = pw * ph;
            amountOfCells = pwph << 5;
            levelWidth = pw << 5;
            secondBuffer = new OpenCLBuffer<uint>(ocl, second);
            patternBuffer = new OpenCLBuffer<uint>(ocl, pattern);

            // pass values to the kernel.
            if (GLInterop)
            {
                kernel.SetArgument(0, image);
                kernel.SetArgument(4, patternBuffer);
                kernel.SetArgument(5, secondBuffer);
            }
            else
            {
                kernel.SetArgument(0, buffer);
            }
            kernel.SetArgument(1, levelWidth);
            kernel.SetArgument(2, ph);
            kernel.SetArgument(3, amountOfCells);
            kernel.SetArgument(8, screenWidth);
            kernel.SetArgument(9, screenHeight);

            // pass values to the copy kernel
            copyKernel.SetArgument(0, levelWidth);
            copyKernel.SetArgument(1, amountOfCells);
            copyKernel.SetArgument(2, patternBuffer);
            copyKernel.SetArgument(3, secondBuffer);
        }
        // TICK
        // Main application entry point: the template calls this function once per frame.
        public void Tick()
        {
            // start timer
            timer.Restart();
            // run the simulation, 1 step
            GL.Finish();
	        // clear the screen
	        screen.Clear(0);
            // do opencl stuff
            if (!GLInterop)
            {
                kernel.SetArgument(0, buffer);
                // if we have an even generation, we set the "pattern" variable in opencl 
                // to the patternBuffer and "second" to the secondBuffer
                // if the generation is odd, we set "pattern" to secondBuffer and
                // "second" to patternBuffer
                // So every generation these buffers get swapped, so we don't have to
                // manually swap them every time
                if (generation % 2 == 0)
                {
                    kernel.SetArgument(4, patternBuffer);
                    kernel.SetArgument(5, secondBuffer);
                }
                else
                {
                    kernel.SetArgument(4, secondBuffer);
                    kernel.SetArgument(5, patternBuffer);
                }
            }
            kernel.SetArgument(6, xoffset);
            kernel.SetArgument(7, yoffset);

            // execute kernel
            long[] workSize = { amountOfCells };
	        if (GLInterop)
	        {
		        // INTEROP PATH:
		        // Use OpenCL to fill an OpenGL texture; this will be used in the
		        // Render method to draw a screen filling quad. This is the fastest
		        // option, but interop may not be available on older systems.
		        // lock the OpenGL texture for use by OpenCL
		        kernel.LockOpenGLObject( image.texBuffer );
		        // execute the kernel
		        kernel.Execute( workSize, null );
                // Wait for the kernel to finish
                kernel.Finish();
                // execute the copy kernel
                copyKernel.Execute(workSize, null);
                // wait for the copykernel to finish
                copyKernel.Finish();
		        // unlock the OpenGL texture so it can be used for drawing a quad
		        kernel.UnlockOpenGLObject( image.texBuffer );
	        }
	        else
	        {
		        // NO INTEROP PATH:
		        // Use OpenCL to fill a C# pixel array, encapsulated in an
		        // OpenCLBuffer<int> object (buffer). After filling the buffer, it
		        // is copied to the screen surface, so the template code can show
		        // it in the window.
		        // execute the kernel
		        kernel.Execute( workSize, null );
                // get the data from the device to the host
                buffer.CopyFromDevice();
                // plot pixels using the data on the host
                for (int y = 0; y < screenHeight; y++)
                {
                    for (int x = 0; x < screenWidth; x++)
                    {
                        int index = x + y * screenWidth;
                        screen.pixels[index] = buffer[index];
                    }
                }
	        }
            // report performance
            Console.WriteLine("generation " + generation++ + ": " + timer.ElapsedMilliseconds + "ms");
        }
        public void Render() 
        {
	        // use OpenGL to draw a quad using the texture that was filled by OpenCL
	        if (GLInterop)
	        {
		        GL.LoadIdentity();
                GL.Scale(zoom, zoom, 1.0f);
                GL.BindTexture( TextureTarget.Texture2D, image.OpenGLTextureID );
		        GL.Begin( PrimitiveType.Quads );
                GL.TexCoord2(0.0f, 1.0f); GL.Vertex2(-1.0f, -1.0f);
		        GL.TexCoord2(1.0f, 1.0f); GL.Vertex2(1.0f, -1.0f);
		        GL.TexCoord2(1.0f, 0.0f); GL.Vertex2(1.0f, 1.0f);
		        GL.TexCoord2(0.0f, 0.0f); GL.Vertex2(-1.0f, 1.0f);
                GL.End();
            }
        }

        // Check for user keyboard input to zoom in the camera
        public void ZoomCamera()
        {
            var keyboard = OpenTK.Input.Keyboard.GetState();
            if (keyboard[OpenTK.Input.Key.Plus] || keyboard[OpenTK.Input.Key.KeypadPlus])
            {
                /*if (zoom <= 2)
                {
                    zoom += 1f;
                }*/
                zoom *= 2;
            }

            else if (keyboard[OpenTK.Input.Key.Minus] || keyboard[OpenTK.Input.Key.KeypadMinus])
            {
                /*if (zoom >= 1)
                {
                    zoom -= 1f;
                }*/
                zoom *= 0.05f;
            }
        }
    }

} // namespace Template