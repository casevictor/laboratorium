from __future__ import print_function
import pyopencl as cl
import sys
from PIL import Image, ImageFilter
import numpy
import datetime
from time import time
import timeit

def CreateProgram(context, device, fileName):
    kernelFile = open(fileName, 'r')
    kernelStr = kernelFile.read()
    program = cl.Program(context, kernelStr).build()
    return program

def LoadImage(context, fileName):
    im = Image.open(fileName)
    if im.mode != "RGBA":
        im = im.convert("RGBA")

    buffer = im.tostring()
    clImageFormat = cl.ImageFormat(cl.channel_order.RGBA,cl.channel_type.UNORM_INT8)

    clImage = cl.Image(context,
                       cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                       clImageFormat,
                       im.size,
                       None,
                       buffer
                       )
    
    return clImage, im.size

def SaveImage(fileName, buffer, imgSize):
    im = Image.fromstring("RGBA", imgSize, buffer.tostring())
    im.save(fileName)

def RoundUp(groupSize, globalSize):
    r = globalSize % groupSize;
    if r == 0:
        return globalSize;
    else:
        return globalSize + groupSize - r;

def referenceFilter():
    image = Image.open(sys.argv[1])
    image = image.filter(ImageFilter.FIND_EDGES)
    image.save('new_name.png')

def main():

    if len(sys.argv) != 2:
        print("USAGE: " + sys.argv[0] + " <inputImageFile>")
        return 1
    
    print("Step One: Executing without OpenCL")
    print("Executed program succesfully. %g s" % min(timeit.Timer(referenceFilter).repeat(repeat=1, number=1)))
    
    
    print("Step Two: Executing with OpenCL")
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            print("===============================================================")
            print("Platform name:", platform.name)
            print("Platform profile:", platform.profile)
            print("Platform vendor:", platform.vendor)
            print("Platform version:", platform.version)
            print("---------------------------------------------------------------")
            print("Device name:", device.name)
            print("Device type:", cl.device_type.to_string(device.type))
            print("Device memory: ", device.global_mem_size//1024//1024, 'MB')
            print("Device max clock speed:", device.max_clock_frequency, 'MHz')
            print("Device compute units:", device.max_compute_units)
            print("Device max work group size:", device.max_work_group_size)
            print("Device max work item sizes:", device.max_work_item_sizes)
        
            imageObjects = [ 0, 0 ]
            context = cl.Context([device])
            commandQueue = cl.CommandQueue(context, device,properties=cl.command_queue_properties.PROFILING_ENABLE)
    
        # Make sure the device supports images, otherwise exit
            if not device.get_info(cl.device_info.IMAGE_SUPPORT):
                print("OpenCL device does not support images.")
                return 1

            # Load input image from file and load it into
            # an OpenCL image object
            imageObjects[0], imgSize = LoadImage(context, sys.argv[1])
    
            # Create ouput image object
            clImageFormat = cl.ImageFormat(cl.channel_order.RGBA,cl.channel_type.UNORM_INT8)
            imageObjects[1] = cl.Image(context,cl.mem_flags.WRITE_ONLY,clImageFormat,imgSize)

            # Create sampler for sampling image object
            sampler = cl.Sampler(context,
                    False, #  Non-normalized coordinates
                    cl.addressing_mode.CLAMP_TO_EDGE,
                    cl.filter_mode.NEAREST)
    
            # Create OpenCL program
            program = CreateProgram(context, device, "xFilter.cl")
            if(cl.device_type.to_string(device.type)=="CPU"):
                worker = 2**7
                localWorkSize = (worker, 1)
            else:
                worker = 2**4
                localWorkSize = (worker, worker)

            print(localWorkSize);
            globalWorkSize = (RoundUp(localWorkSize[0], imgSize[0]),RoundUp(localWorkSize[1], imgSize[1]) )
            print(globalWorkSize);

            program.x_filter(commandQueue,
                    globalWorkSize,
                    localWorkSize,
                    imageObjects[0],
                    imageObjects[1],
                    sampler,
                    numpy.int32(imgSize[0]), #width
                    numpy.int32(imgSize[1])  #heigth
            )
                      
            # Read the output buffer back to the Host
            buffer = numpy.zeros(imgSize[0] * imgSize[1] * 4, numpy.uint8)
            origin = ( 0, 0, 0 )
            region = ( imgSize[0], imgSize[1], 1 )
                      
            exec_evt = cl.enqueue_read_image(commandQueue, imageObjects[1],origin, region, buffer)
            exec_evt.wait()
            elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)
            print("Executed program succesfully. %g s" % elapsed)

            # Save the image to disk
            SaveImage("(" + cl.device_type.to_string(device.type) + ")" + sys.argv[1], buffer, imgSize)

main()