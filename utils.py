# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 16:56:48 2022

@author: Sabah
"""

import cplex_methods as cpx
import matplotlib.pyplot as plt  # For plots to be generated
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import os
import cv2
from PIL import Image
import natsort

__all__ = []

def apply_standard_cplex(instance, cost_values, n_items, n_constraints):
    obj_full, sol_full, gap = cpx.solve_using_cplex(instance, cost_values, n_items,n_constraints)
   # print("objective using CPLEX: ", obj_full)
    return obj_full, sol_full, gap


# def save_image_to_folder():
#     files = Common_Methods.generate_file_location()
#     images = Common_Methods.generate_image_location()
    
    

# # define a function which returns an image as numpy array from figure
# def get_img_from_fig(fig, dpi=180):
#     buf = io.BytesIO()
#     fig.figure.savefig(buf, format="png", dpi=dpi)
#     buf.seek(0)
#     img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
#     buf.close()
#     img = cv2.imdecode(img_arr, 1)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.imwrite('cv2_image.png',img)
#     return img

# Video Generating function
def generate_video(image_folder, video_name):
#    string_episode = '\\Episode_' + str(episode)
#    image_folder = generate_image_location(mkp_loc, run_id) + string_episode
#    video_name = 'test.avi'
  #  os.chdir("C:\\Python\\Geekfolder2")
      
    images = [img for img in os.listdir(image_folder)
              if img.endswith(".jpg") or
                 img.endswith(".jpeg") or
                 img.endswith("png")]
     
    #images.sort(key=os.path.getmtime)
   
    images = natsort.natsorted(images)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
  
    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape  
  
    video = cv2.VideoWriter(video_name, 0, 1, (width, height)) 
  
    # Appending the images to the video one by one
    for image in images: 
        video.write(cv2.imread(os.path.join(image_folder, image))) 
      
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    video.release()  # releasing the video generated
    


def create_image_from_matrix(mat, name, objective, feasibility):
    pixel_plot = plt.figure()
    #plt.title("pixel_plot")
    plt.axis('off')
    pixel_plot = plt.imshow(mat, cmap='RdYlGn', interpolation='nearest')
    #   plt.colorbar(pixel_plot)
    # Add text
    plt.text(-2, -2, objective)
    txt = "Z = " + str(objective)
    if feasibility:
        b = {"facecolor": "g", "alpha": 0.5}
    else:
        b = {"facecolor": "r", "alpha": 0.5}

    plt.figtext(0.5, 0.92, txt, ha="center", va="center", fontsize=14, bbox=b)
   # print(name)
    plt.savefig(name)
    plt.close()


def generate_file_location(folder_loc, run_id):
    directory = folder_loc + '\\' + run_id
    os.makedirs(directory, exist_ok = True)
    return directory
    
def generate_image_location(folder_loc, run_id, ep):
    directory = folder_loc + '\\' + run_id +'\\images\\Episode_' +str(ep)
    os.makedirs(directory, exist_ok = True)
    return directory


def generate_graph_location(folder_loc, run_id):
    directory = folder_loc + '\\' + run_id +'\\graphs'
    os.makedirs(directory, exist_ok = True)
    return directory

def generate_video_location(folder_loc, run_id, ep):
    directory = folder_loc + '\\' + run_id +'\\videos\\'
    os.makedirs(directory, exist_ok = True)
    return directory


def generate_graph_of_objectives(training_obj, opt_obj, name):
    y = [opt_obj]*len(training_obj)

    #print("training obj ", training_obj)
    #print(y)

    # plotting the points
    #plt.plot(training_obj, y, color='green', linestyle='dashed', linewidth=3, marker='o', markerfacecolor='blue', markersize=12)
    plt.plot(training_obj, label="Training Objective")
    plt.plot(y, label="Optimal Objective", color='red', linestyle='dashed', linewidth=1.5, marker='o', markerfacecolor='blue', markersize=4)

    # place legend in top right corner
    plt.legend(loc="upper left", mode="expand", ncol=2)


    # setting x and y axis range
   # plt.ylim(1, 8)
   # plt.xlim(1, 8)



    # naming the x axis
    plt.xlabel('training obj')
    # naming the y axis
    plt.ylabel('optimal obj')

    # giving a title to my graph
    plt.title('Objective function value comparison', size=16, fontweight="bold" )

    # function to show the plot
    #plt.show()
    #pixel_plot = plt.imshow()
    plt.savefig(name)
    plt.close()
