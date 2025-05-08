import os
import numpy as np
from matplotlib import pyplot as plt
import io
import wandb
from PIL import Image 
import sys
sys.path.insert(0, os.path.abspath("/data2/laurence220016/Jeff/openvla/"))
from hot3d.hot3d.utils import HOT3D_data_func, load_asset
from Text2HOI.arctic_utils import ARCTIC_data_func
import moviepy
import moviepy.editor as mpy
import torch
stat_dict = {
    'hot3d': {
        'q01': [-0.11492515444755554, -0.10521660327911377, -0.11884290546178818, -0.09759321957826614, -0.11574954211711884, -0.11781377911567688], 
        'q99': [0.1100667107105257, 0.1126265159249307, 0.12030991971492769, 0.09687433362007153, 0.12195313274860388, 0.11489655673503879]
    },
    'arctic': {
        "q01": [-0.06375248245894909, -0.08251506991684437, -0.059924775883555416, -0.0816046864539385, -0.08604897841811181, -0.06955469816923142], 
        "q99": [0.08202865876257426, 0.07596488863229753, 0.05742939833551646, 0.05838170867413283, 0.08410088218748572, 0.06867202304303653],
    }

}
def unnormalize(pred_action, dataset_name):
    stat = stat_dict[dataset_name]
    low = np.array(stat['q01'])
    high = np.array(stat['q99'])
    result_a = (pred_action + 1) * (high - low + 1e-8) / 2.0
    result_a = result_a + low
    return result_a

def fig_to_numpy(fig):
    """Convert a matplotlib figure to a numpy array"""
    # Draw the renderer
    fig.canvas.draw()
    
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    # print('w, h: ', w, h)
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    # The buffer is RGBA (4 channels)
    buf = buf.reshape(h, w, 4)
    # Convert RGBA to RGB
    buf = buf[:, :, :3]
    
    return buf

def render_arctic_pred(collect_batch, pred_actions):
    collect_points = []
    collect_images = []
    collect_gt_points = []
    collect_lang = []
    collect_prev_points = []
    collect_gt_action_points = []
    arctic_data_func = ARCTIC_data_func()
    pred_sequence_name = ''
    for batch, p_a in zip(collect_batch, pred_actions):
        cur_state = batch['state'][0]
        # print('cur_state: ', cur_state)
        
        #print(cur_state, 'jjjj',p_a)
        # print('lsklka batch: ', batch)
        # print('lsklka p_a: ', p_a)
        if isinstance(cur_state, torch.Tensor):
            cur_state = cur_state.cpu().numpy()
        if len(p_a.shape) >= 2:
            p_a = p_a[0]
        next_state = cur_state + unnormalize(p_a.cpu().numpy(),'arctic')
        next_state = next_state.reshape(2, -1)

        # from gt action
        gt_action = batch['gt_action'][0]
        gt_action = unnormalize(gt_action.cpu().numpy(), 'arctic')
        # print('gt_action: ',gt_action)
        #gt_action_next = cur_state[0] + unnormalize(gt_action.cpu().numpy(), 'arctic')
        gt_action_next = cur_state + gt_action
        gt_action_next = gt_action_next.reshape(2, -1)


        # print('next_state: ', next_state.shape)
        sequence_name = batch['sequence_name'][0]
        if isinstance(sequence_name, bytes):
            sequence_name = sequence_name.decode('utf-8')
        elif not isinstance(sequence_name, str):
            sequence_name = str(sequence_name)
        timestamp = batch['timestamp_ns'][0]
       
        
        next_timestamp = batch['next_timestamp_ns'][0]
        if sequence_name != pred_sequence_name:
            pred_sequence_name = sequence_name
            arctic_data_func.set_sequence(sequence_name)
        
        lhand_joint, rhand_joint = arctic_data_func.get_3d_joints(timestamp)
        # print('cur data_state:',[lhand_joint[0], rhand_joint[0]]) 
        n_lhand_joint, n_rhand_joint = arctic_data_func.get_3d_joints(next_timestamp)
        # print('next data_state:',[n_lhand_joint[0], n_rhand_joint[0]])
        # print('gt_action_next: ', [n_lhand_joint[0]-lhand_joint[0] , n_rhand_joint[0]-rhand_joint[0]])

        next_lhand_joint_2d, next_rhand_joint_2d = arctic_data_func.get_2d_joints(next_timestamp)
        prev_lhand_joint_2d, prev_rhand_joint_2d = arctic_data_func.get_2d_joints(timestamp)
        pred_points_2d = arctic_data_func.transform_camera_to_2d(next_state)

        gt_action_points_2d = arctic_data_func.transform_camera_to_2d(gt_action_next)
        collect_gt_action_points.append(gt_action_points_2d)


        image = arctic_data_func.get_image(next_timestamp)
        collect_points.append(pred_points_2d)
        collect_images.append(image)
        collect_gt_points.append([next_lhand_joint_2d[0], next_rhand_joint_2d[0]])
        collect_prev_points.append([prev_lhand_joint_2d[0], prev_rhand_joint_2d[0]])
        # collect_lang.append(batch['lang'][0])

    freq = 1
    render_images = []
    collect_points = np.array(collect_points)
    collect_gt_points = np.array(collect_gt_points)
    collect_gt_action_points = np.array(collect_gt_action_points)
    # print('collect_points: ', collect_points.shape)
    # print('collect_gt_points: ', collect_gt_points.shape)
    collect_prev_points = np.array(collect_prev_points)
    for i in range(0, len(collect_images), freq):
        plt.clf()
        fig = plt.figure()
        plt.imshow(collect_images[i])
        plt.scatter(collect_points[i][:, 0], collect_points[i][:, 1], color='red', s=30)
        plt.scatter(collect_gt_points[i][:, 0], collect_gt_points[i][:, 1], color='blue', s=20)
        #plt.scatter(collect_gt_action_points[i][:, 0], collect_gt_action_points[i][:, 1], color='green', s=20)
        # plot gt action
        # x = np.array([collect_prev_points[i][0][0], collect_gt_points[i][0][0]])
        # y = np.array([collect_prev_points[i][0][1], collect_gt_points[i][0][1]])
        # plt.plot(x, y, color='blue', linestyle='-', linewidth=2, marker='o', markersize=6)
        #  # plot pred action
        # x = np.array([collect_prev_points[i][0][0], collect_points[i][0][0]])
        # y = np.array([collect_prev_points[i][0][1], collect_points[i][0][1]])
        #plt.plot(x, y, color='red', linestyle='-', linewidth=2, marker='o', markersize=6)
        
        # plt.title(f"{collect_lang[i]} - Red: Predicted, Blue: Ground Truth")
        plt.show()
        # Convert figure to numpy array instead of displaying it
        img_array = fig_to_numpy(fig)
        plt.close(fig)  # Close the figure to free memory
        render_images.append(img_array)

    clip = mpy.ImageSequenceClip(render_images, fps=2)
    clip.write_videofile(f"video/test_video.mp4")





def render_pred_wrist_translation(collect_batch, pred_actions, frame='world', fps=30):
    object_lib_path = './hot3d/hot3d/dataset/assets/'
    mano_path = './hot3d/mano_v1_2/models/'
    object_library, mano_model = load_asset(object_lib_path, mano_path)
    


    for batch , p_a in zip(collect_batch, pred_actions):
        cur_state = batch['state'][0]
        #print(cur_state, 'jjjj',p_a)
        
        next_state = cur_state[0] + unnormalize(p_a.cpu().numpy(),'hot3d')

        next_state_gt = cur_state[0] + batch['gt_action'].cpu().numpy() /10.0

        #print(next_state)
        next_state = next_state.reshape(2, -1)
        #print(next_state)
        sequence_name = batch['sequence_name'][0].decode()
        sequence_path = os.path.join('./hot3d/hot3d/dataset/', sequence_name)
        timestamp_ns = batch['timestamp_ns'][0]

        next_timestamp_ns = batch['next_timestamp_ns'][0]

        hot3d_data_func = HOT3D_data_func(sequence_path, object_library, mano_model)  

        #next_timestamp_ns = hot3d_data_func.get_next_timestamp(timestamp_ns, fps)

        next_rgb = hot3d_data_func.get_rgb_camera_data(next_timestamp_ns)
        print('frame', frame)
        point_2d_list = hot3d_data_func.transform_3d_to_2d(next_timestamp_ns, next_state, frame)

        cur_hand_data = hot3d_data_func.get_hand_pose_data(timestamp_ns)

        cur_hand_data_2d = hot3d_data_func.transform_3d_to_2d(timestamp_ns,  [cur_hand_data['left_wrist'], cur_hand_data['right_wrist']]) 
        cur_hand_data_c = hot3d_data_func.transform_world_to_camera(timestamp_ns, [cur_hand_data['left_wrist'], cur_hand_data['right_wrist']])
        cur_hand_data_c = np.concatenate(cur_hand_data_c)
        print('cur hand', cur_hand_data_c)
        print('cur_state', cur_state)

        gt_hand_data = hot3d_data_func.get_hand_pose_data(next_timestamp_ns)
        gt_hand_wrist = [gt_hand_data['left_wrist'], gt_hand_data['right_wrist']]
        gt_2d_list = hot3d_data_func.transform_3d_to_2d(next_timestamp_ns, gt_hand_wrist) 

        gt_list_camera = hot3d_data_func.transform_world_to_camera(next_timestamp_ns, gt_hand_wrist) 
        print('=='*20)
        print('pred')
        print(next_state)
        print('batch gt')
        print(next_state_gt)
        print('hot3d gt')
        gt_list_camera = np.concatenate(gt_list_camera)
        print(gt_list_camera)
        print('data action')
        print(gt_list_camera - cur_hand_data_c)
        print('detoken action')
        print(batch['gt_action'].cpu().numpy())

        print('=='*20)

        plt.clf()
        plt.figure()
        plt.imshow(next_rgb['image'], interpolation="nearest")
        plt.scatter(
            x=[x[0] for x in gt_2d_list if x is not None],
            y=[x[1] for x in gt_2d_list if x is not None],
            s=30,
            c="b",
        )

        for a, b in zip(cur_hand_data_2d, gt_2d_list):
            if a is not None and b is not None:
                plt.plot([a[0], b[0]], [a[1], b[1]], color='blue', linewidth=2)

        plt.scatter(
            x=[x[0] for x in point_2d_list if x is not None],
            y=[x[1] for x in point_2d_list if x is not None],
            s=30,
            c="r",
        )

        if wandb.run is not None:
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)

            pil_image = Image.open(buffer)
            pil_image = pil_image.rotate(270, expand=True)  
            # Log the image to WandB
            wandb.log({"plot": wandb.Image(pil_image)})



        #print('2D: ', point_2d)
        #print('3D: ', next_state)


