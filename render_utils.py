import os
import numpy as np
from matplotlib import pyplot as plt
import io
import wandb
from PIL import Image 
import sys
sys.path.insert(0, os.path.abspath("/home/bidex/dex_openvla/"))
#from hot3d.hot3d.utils import HOT3D_data_func, load_asset
from Text2HOI.arctic_utils import ARCTIC_data_func
from Text2HOI.h2o_utils import h2o_data_func
import moviepy
import moviepy as mpy
import torch
from hoi4d_utils import Hoi4d_data_func
from oakink2_utils import OakInk2DataFunc
from taco_utils import TacoDataFunc
from humanpolicy_utils import HumanpolicyDataFunc
from einops import rearrange

import plotly.graph_objects as go
from plotly.subplots import make_subplots





stat_dict = {
    'hot3d': {
        'q01': [-0.11492515444755554, -0.10521660327911377, -0.11884290546178818, -0.09759321957826614, -0.11574954211711884, -0.11781377911567688], 
        'q99': [0.1100667107105257, 0.1126265159249307, 0.12030991971492769, 0.09687433362007153, 0.12195313274860388, 0.11489655673503879]
    },
    'arctic': {
        "q01": [-0.06375248245894909, -0.08251506991684437, -0.059924775883555416, -0.0816046864539385, -0.08604897841811181, -0.06955469816923142], 
        "q99": [0.08202865876257426, 0.07596488863229753, 0.05742939833551646, 0.05838170867413283, 0.08410088218748572, 0.06867202304303653],
    },
    'h2o': {
        "q01": [-0.027184246480464934, -0.04457992743700743, -0.0407070517539978, -0.04403566308319569, -0.04243455901741981, -0.05199018269777298], 
        "q99": [0.02747734859585746, 0.03982493728399275, 0.0338872045278548, 0.03982752002775669, 0.05112806670367717, 0.050409896671772],
    },
    'hoi4d': {
        # "q01": [-0.025924417674541473, -0.045511924773454664, -0.03003883481025696, -0.03323337104171514, -0.051958327293395994, -0.04170384883880615], 
        # "q99": [0.027119120657444016, 0.04241098582744603, 0.03212092250585559, 0.03761008340865374, 0.05233722805976874, 0.04073245912790301]
        "q01": [0.0, 0.0, 0.0, -0.035353663563728335, -0.06155168622732163, -0.042915129363536836], 
        "q99": [0.0, 0.0, 0.0, 0.035711939334869526, 0.06349327445030228, 0.04803465455770502]
    }, 
    'oakink2': {
       "q01": [-0.04848216846585274, -0.038360562175512314, -0.04476017504930496, -0.07783905416727066, -0.05707544833421707, -0.060008201748132706], 
        "q99": [0.05282054468989372, 0.05441252887248993, 0.03324000909924507, 0.07287413626909256, 0.06427331268787384, 0.04736606776714325]
    },
    'oakink2-finger': {
       "q01": [-0.04848216846585274, -0.038360562175512314, -0.04476017504930496, -0.07783905416727066, -0.05707544833421707, -0.060008201748132706], 
       "q99": [0.05282054468989372, 0.05441252887248993, 0.03324000909924507, 0.07287413626909256, 0.06427331268787384, 0.04736606776714325],
    },

    # 'oakink2': {
    #     "q01": [-0.04363662376999855, -0.039132095873355865, -0.037270285189151764, -0.053785186260938644, -0.04852518066763878, -0.05175589397549629], 
    #     "q99": [0.0411435104906559, 0.044969361275434494, 0.030560173094272614, 0.052058082073926926, 0.04941689968109131, 0.0431147962808609]
    # }, 
    'taco': {
       "q01": [-0.055475939437747, -0.07929554991424083, -0.05178946554660797, -0.06250557545572519, -0.07754244767129422, -0.07249288707971573], 
       "q99": [0.05526134632527816, 0.07532032988965508, 0.04736755099147551, 0.06374433524906635, 0.07931515499949444, 0.07170739397406559]
    }, 'humanpolicy': {
       "q01": [0.0, 0.0, 0.0, -0.07219996556639671, -0.116965871155262, -0.18842112243175507], 
       "q99": [0.0, 0.0, 0.0, 0.070285872519016, 0.12452144220471373, 0.13730108141899072],
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

def transf_point_array_np(transf: np.ndarray, point: np.ndarray):
    # transf: [..., 4, 4]
    # point: [..., X, 3]
    leading_shape = point.shape[:-2]
    leading_dim = len(leading_shape)

    res = (
        np.swapaxes(
            np.matmul(
                transf[..., :3, :3],
                np.swapaxes(point, leading_dim, leading_dim + 1),
            ),
            leading_dim,
            leading_dim + 1,
        )  # [..., X, 3]
        + transf[..., :3, 3][..., np.newaxis, :]  # [..., 1, 3]
    )
    return res

def render_pred(collect_batch, gt_actions, pred_actions, dataset_name, action_chunk_size=1):
    collect_points = []
    collect_images = []
    collect_gt_points = []
    collect_lang = []
    collect_prev_points = []
    collect_gt_action_points = []
    if dataset_name == 'arctic':
        arctic_data_func = ARCTIC_data_func()
    elif dataset_name == 'h2o':
        arctic_data_func = h2o_data_func()
    elif dataset_name == 'hoi4d':
        arctic_data_func = Hoi4d_data_func()
    elif 'oakink2' in dataset_name:
        arctic_data_func = OakInk2DataFunc(dataset_root='/mnt/lab-tank/fang/OakInk2_data/', mano_path='/home/bidex/dex_openvla/OakInk2/asset/mano_v1_2')
    elif dataset_name == 'taco':
        arctic_data_func = TacoDataFunc(dataset_root='/mnt/lab-tank/fang/taco_data/', object_model_root='/mnt/lab-tank/fang/taco_data/object_models_released')
    elif dataset_name == 'humanpolicy':
        arctic_data_func = HumanpolicyDataFunc()
    pred_sequence_name = ''
    is_right_hand = True
    is_left_hand = True
    text_list = []
    is_left_hand_list = []
    is_right_hand_list = []
    pred_action_lens = []
    gt_action_lens = []

    joint_idx = [0, 2, 5, 9, 13, 17] 

    a_c_size = action_chunk_size # 1~4
    for batch, gt_a_list, p_a_list in zip(collect_batch, gt_actions, pred_actions):


        sequence_name = batch['sequence_name'][0]
        if isinstance(sequence_name, bytes):
            sequence_name = sequence_name.decode('utf-8')
        elif not isinstance(sequence_name, str):
            sequence_name = str(sequence_name)
        timestamp = batch['timestamp_ns'][0]
        print('current t: ', timestamp)
        extrinsic = batch['extrinsic'][0]
        # print('batch extrinsic: ', extrinsic)
        
        print('extrinsic', extrinsic)
       
        
        next_timestamp = batch['next_timestamp_ns'][0]
        print('next t: ', next_timestamp)
        if sequence_name != pred_sequence_name:
            pred_sequence_name = sequence_name
            arctic_data_func.set_sequence(sequence_name, timestamp)
        print('dataset extrinsic: ', arctic_data_func.get_extrinsic(timestamp))

        cur_state = batch['state'][0]
        # print('cur_state: ', cur_state)
        
        #print(cur_state, 'jjjj',p_a)
        # print('lsklka batch: ', batch)
        # print('lsklka p_a: ', p_a)
        if isinstance(cur_state, torch.Tensor):
            cur_state = cur_state.cpu().numpy()
            
        # if len(p_a.shape) >= 2:
        #     p_a = p_a[0]    # only use the first action
        dataset_action = arctic_data_func.get_action(timestamp, next_timestamp)

        print('gt batch action shape', gt_a_list.shape)
        
        a_idx = np.arange(a_c_size)
        p_a_list = p_a_list[a_idx]
        gt_action_list = gt_a_list[a_idx]
        
        p_a_2d_list = []
        gt_a_2d_list = []
        p_cur_state = cur_state
        print('batch, state', cur_state)
        c_ljoints, c_rjoints = arctic_data_func.get_3d_joints(timestamp)
        print('datset state: ', c_ljoints[joint_idx], c_rjoints[joint_idx])
        g_cur_state = cur_state
        
        gt_action_list = dataset_action

        print('ori action: ', batch['ori_action'][0])
        

        for p_a, gt_a in zip(p_a_list, gt_action_list):

            # pred_action = unnormalize(p_a.cpu().numpy(),dataset_name)
            gt_a = gt_a[joint_idx].reshape(-1)
            print('batch action: ', gt_a_list)
            print('dataset action: ', gt_a)
            print('pred action: ', p_a)
            next_state = p_cur_state + p_a.cpu().numpy() #
            p_cur_state = next_state
            num_points = next_state.shape[0] // 3
            next_state = next_state.reshape(num_points, 3)  # camera frame at T=t
            next_state = transf_point_array_np(np.linalg.inv(extrinsic), next_state) #world frame
            next_extrinsic = arctic_data_func.get_extrinsic(next_timestamp)
            next_state = transf_point_array_np(next_extrinsic, next_state) #camera frame at T=t+1

            # from gt action
            # gt_action = unnormalize(gt_a.cpu().numpy(), dataset_name)


            # gt_action_next = g_cur_state + gt_a.cpu().numpy()
            # g_cur_state = gt_action_next
            # num_points = gt_action_next.shape[0] // 3
            # gt_action_next = gt_action_next.reshape(num_points, 3)
            # gt_action_next = transf_point_array_np(np.linalg.inv(extrinsic), gt_action_next)


            n_lhand_joint, n_rhand_joint = arctic_data_func.get_world_3d_joints(next_timestamp)
            # print('shape', n_lhand_joint.shape)
            gt_action_next = np.concatenate([n_lhand_joint[np.array(joint_idx)], n_rhand_joint[np.array(joint_idx)]])
            
            gt_action_next = transf_point_array_np(next_extrinsic, gt_action_next)

            pred_points_2d = arctic_data_func.transform_camera_to_2d(next_state)
            # print('nnnnn', pred_points_2d.shape)
            gt_action_points_2d = arctic_data_func.transform_camera_to_2d(gt_action_next)
            # print('nnnnnk', gt_action_points_2d.shape)
            p_a_2d_list.append(pred_points_2d)
            gt_a_2d_list.append(gt_action_points_2d)
        
        p_a_2d_list = np.stack(p_a_2d_list)   # (a_c_size, 2, 2)
        gt_a_2d_list = np.stack(gt_a_2d_list)
        
        # print('88888', p_a_2d_list.shape)
        # print('999999', gt_a_2d_list.shape)

        collect_gt_action_points.append(gt_a_2d_list)
  
        collect_points.append(p_a_2d_list)   

        # flat_pred_action = pred_action.reshape(-1)
        # flat_gt_action = gt_action.reshape(-1)
        # pred_action_lens.append(np.linalg.norm(flat_pred_action))
        # gt_action_lens.append(np.linalg.norm(flat_gt_action))

        # print('next_state: ', next_state.shape)
        
        
        n_lhand_joint, n_rhand_joint = arctic_data_func.get_3d_joints(next_timestamp)
        # print('next data_state:',[n_lhand_joint[0], n_rhand_joint[0]])
        # print('gt_action_next: ', [n_lhand_joint[0]-lhand_joint[0] , n_rhand_joint[0]-rhand_joint[0]])
        is_left_hand = n_lhand_joint[0].any()
        is_right_hand = n_rhand_joint[0].any()
        is_left_hand_list.append(is_left_hand)
        is_right_hand_list.append(is_right_hand)
        # print('is_left_hand', is_left_hand)
        # print('is_right_hand', is_right_hand)
        # print('n hand', n_lhand_joint[0], n_rhand_joint[0])
        # print('next state', next_state)
        next_lhand_joint_2d, next_rhand_joint_2d = arctic_data_func.get_2d_joints(next_timestamp)
        #print('get_2d_joints', next_lhand_joint_2d.shape, next_rhand_joint_2d.shape)
        prev_lhand_joint_2d, prev_rhand_joint_2d = arctic_data_func.get_2d_joints(timestamp)
        
        # print('pred_points_2d', pred_points_2d)
        # print('next_lhand_joint_2d', next_lhand_joint_2d[0], next_rhand_joint_2d[0])
        
        
        image = arctic_data_func.get_image(next_timestamp)
        
        collect_images.append(image)
        collect_gt_points.append([next_lhand_joint_2d[0], next_rhand_joint_2d[0]])
        collect_prev_points.append([prev_lhand_joint_2d[0], prev_rhand_joint_2d[0]])
        # text_list.append(arctic_data_func.get_instruction())
        # collect_lang.append(batch['lang'][0])

    freq = 1
    render_images = []
    collect_points = np.array(collect_points)
    # print('ccollkj', collect_points.shape)
    collect_gt_points = np.array(collect_gt_points)
    collect_gt_action_points = np.array(collect_gt_action_points)
    # print('collect_points: ', collect_points.shape)
    # print('collect_gt_points: ', collect_gt_points.shape)
    collect_prev_points = np.array(collect_prev_points)
    m = 100
    
    h, w, _ = collect_images[0].shape

    for i in range(0, len(collect_images), freq):

        render_hands = []
        if is_left_hand_list[i]:
            render_hands.append(0)
        if is_right_hand_list[i]:
            render_hands.append(1)
        render_hands = np.array(render_hands)

        plt.clf()
        fig = plt.figure()
        plt.imshow(collect_images[i])

        # Helper function to filter points
        def filter_points(points):
            mask = (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
            return points[mask], mask

        # Filter and plot collect_points
        points = collect_points[i][:, :]
        # print('points', points.shape)
        points = points.reshape(-1, 2)
        points_f, mask_points = filter_points(points)
        plt.scatter(points_f[:, 0], points_f[:, 1], color='red', s=30)

        # Filter and plot collect_gt_points
        gt_points = collect_gt_points[i][render_hands]
        gt_points_f, mask_gt = filter_points(gt_points)
        plt.scatter(gt_points_f[:, 0], gt_points_f[:, 1], color='blue', s=20)

        # Filter and plot collect_gt_action_points
        gt_action_points = collect_gt_action_points[i][:, :]
        gt_action_points = gt_action_points.reshape(-1, 2)
        gt_action_points_f, mask_gt_action = filter_points(gt_action_points)
        plt.scatter(gt_action_points_f[:, 0], gt_action_points_f[:, 1], color='green', s=30)

        # for idx in range(len(gt_action_points_f)-3):
        #     p0 = gt_action_points_f[idx+1]
        #     p1 = gt_action_points_f[idx+3]
        #     if (0 <= p0[0] < w and 0 <= p0[1] < h and 0 <= p1[0] < w and 0 <= p1[1] < h):
        #         x = np.array([p0[0], p1[0]])
        #         y = np.array([p0[1], p1[1]])
        #         plt.plot(x, y, color='green', linestyle='-', linewidth=2)



        # Filter and plot collect_prev_points
        prev_points = collect_prev_points[i][render_hands]
        prev_points_f, mask_prev = filter_points(prev_points)
        plt.scatter(prev_points_f[:, 0], prev_points_f[:, 1], color='yellow', s=30)

        # # Plot lines only if both endpoints are in bounds
        # for idx in range(len(render_hands)):
        #     # For gt line
        #     p0 = prev_points[idx]
        #     p1 = gt_points[idx]
        #     if (0 <= p0[0] < w and 0 <= p0[1] < h and 0 <= p1[0] < w and 0 <= p1[1] < h):
        #         x = np.array([p0[0], p1[0]])
        #         y = np.array([p0[1], p1[1]])
        #         plt.plot(x, y, color='green', linestyle='-', linewidth=2)

        # # plot pred action
        # x = np.array([prev_points_f[0][0], points_f[0][0]])
        # y = np.array([prev_points_f[0][1], points_f[0][1]])
        # plt.plot(x, y, color='red', linestyle='-', linewidth=2, marker='o', markersize=6)
        
        # plt.title(f"{collect_lang[i]} - Red: Predicted, Blue: Ground Truth")
        # plt.title(f"{text_list[i]}")
        plt.show()
        # Convert figure to numpy array instead of displaying it
        img_array = fig_to_numpy(fig)
        plt.close(fig)  # Close the figure to free memory
        render_images.append(img_array)

    clip = mpy.ImageSequenceClip(render_images, fps=2)
    clip.write_videofile(f"/home/bidex/dex_spatialvla/video/test_video_{dataset_name}.mp4")
    

    # print('pred_action_lens', sum(pred_action_lens)/len(pred_action_lens))
    # print('gt_action_lens', sum(gt_action_lens)/len(gt_action_lens))

def render_3d_pred2(collect_batch, gt_actions, pred_actions, dataset_name, action_chunk_size=1):
    """
    Render 3D animation of prediction results similar to render_pred but in 3D space
    """
    # Initialize data collection lists
    collect_3d_points = []
    collect_3d_gt_points = []
    collect_3d_prev_points = []
    collect_3d_gt_action_points = []
    collect_head_points = []
    collect_timestamps = []
    
    # Initialize dataset function
    if dataset_name == 'arctic':
        from Text2HOI.arctic_utils import ARCTIC_data_func
        arctic_data_func = ARCTIC_data_func()
    elif dataset_name == 'h2o':
        from Text2HOI.h2o_utils import h2o_data_func
        arctic_data_func = h2o_data_func()
    elif dataset_name == 'hoi4d':
        from hoi4d_utils import Hoi4d_data_func
        arctic_data_func = Hoi4d_data_func()
    elif 'oakink2' in dataset_name:
        from oakink2_utils import OakInk2DataFunc
        arctic_data_func = OakInk2DataFunc(dataset_root='/mnt/lab-tank/fang/OakInk2_data/', 
                                          mano_path='/home/bidex/dex_openvla/OakInk2/asset/mano_v1_2')
    elif dataset_name == 'taco':
        from taco_utils import TacoDataFunc
        arctic_data_func = TacoDataFunc(dataset_root='/mnt/lab-tank/fang/taco_data/', 
                                       object_model_root='/mnt/lab-tank/fang/taco_data/object_models_released')
    elif dataset_name == 'humanpolicy':
        from humanpolicy_utils import HumanpolicyDataFunc
        arctic_data_func = HumanpolicyDataFunc()
    
    pred_sequence_name = ''
    joint_idx = [0, 2, 5, 9, 13, 17]
    a_c_size = action_chunk_size
    
    # Helper function from render_pred
    def transf_point_array_np(transf: np.ndarray, point: np.ndarray):
        leading_shape = point.shape[:-2]
        leading_dim = len(leading_shape)
        res = (
            np.swapaxes(
                np.matmul(
                    transf[..., :3, :3],
                    np.swapaxes(point, leading_dim, leading_dim + 1),
                ),
                leading_dim,
                leading_dim + 1,
            )
            + transf[..., :3, 3][..., np.newaxis, :]
        )
        return res
    
    print("Collecting 3D data for animation...")
    
    # Step 1: Collect important data
    for batch, gt_a_list, p_a_list in zip(collect_batch, gt_actions, pred_actions):
        sequence_name = batch['sequence_name'][0]
        if isinstance(sequence_name, bytes):
            sequence_name = sequence_name.decode('utf-8')
        elif not isinstance(sequence_name, str):
            sequence_name = str(sequence_name)
            
        timestamp = batch['timestamp_ns'][0]
        next_timestamp = batch['next_timestamp_ns'][0]
        extrinsic = batch['extrinsic'][0]
        
        if sequence_name != pred_sequence_name:
            pred_sequence_name = sequence_name
            arctic_data_func.set_sequence(sequence_name, timestamp)
        
        cur_state = batch['state'][0]
        if isinstance(cur_state, torch.Tensor):
            cur_state = cur_state.cpu().numpy()
        
        dataset_action = arctic_data_func.get_action(timestamp, next_timestamp)
        a_idx = np.arange(a_c_size)
        p_a_list = p_a_list[a_idx]
        gt_action_list = gt_a_list[a_idx]
        gt_action_list = dataset_action
        
        # Process actions for this batch
        p_a_3d_list = []
        gt_a_3d_list = []
        p_cur_state = cur_state
        g_cur_state = cur_state
        
        for p_a, gt_a in zip(p_a_list, gt_action_list):
            gt_a = gt_a[joint_idx].reshape(-1)
            
            # Predicted next state in 3D world coordinates
            next_state = p_cur_state + p_a.cpu().numpy()
            p_cur_state = next_state
            num_points = next_state.shape[0] // 3
            next_state_3d = next_state.reshape(num_points, 3)
            # Transform to world frame
            next_state_3d = transf_point_array_np(np.linalg.inv(extrinsic), next_state_3d)
            
            # GT next state in 3D world coordinates
            gt_action_next = g_cur_state + gt_a.cpu().numpy()
            g_cur_state = gt_action_next
            gt_action_next_3d = gt_action_next.reshape(num_points, 3)
            # Transform to world frame
            gt_action_next_3d = transf_point_array_np(np.linalg.inv(extrinsic), gt_action_next_3d)
            
            p_a_3d_list.append(next_state_3d)
            gt_a_3d_list.append(gt_action_next_3d)
        
        p_a_3d_list = np.stack(p_a_3d_list)
        gt_a_3d_list = np.stack(gt_a_3d_list)
        
        collect_3d_points.append(p_a_3d_list)
        collect_3d_gt_action_points.append(gt_a_3d_list)
        
        # Get 3D ground truth joints
        n_lhand_joint, n_rhand_joint = arctic_data_func.get_world_3d_joints(next_timestamp)
        prev_lhand_joint, prev_rhand_joint = arctic_data_func.get_world_3d_joints(timestamp)
        
        collect_3d_gt_points.append([n_lhand_joint[0], n_rhand_joint[0]])
        collect_3d_prev_points.append([prev_lhand_joint[0], prev_rhand_joint[0]])
        
        # Add head position (assuming it's available or can be estimated)
        # For now, we'll use a fixed head position relative to hands
        head_pos = np.mean([n_lhand_joint[0], n_rhand_joint[0]], axis=0) + np.array([0, 0, 0.3])
        collect_head_points.append(head_pos)
        
        collect_timestamps.append(timestamp)
    
    # Convert to numpy arrays
    collect_3d_points = np.array(collect_3d_points)
    collect_3d_gt_points = np.array(collect_3d_gt_points)
    collect_3d_prev_points = np.array(collect_3d_prev_points)
    collect_3d_gt_action_points = np.array(collect_3d_gt_action_points)
    collect_head_points = np.array(collect_head_points)
    
    print(f"Collected {len(collect_3d_points)} frames of 3D data")
    
    # Step 2: Create 3D figure and trace points
    fig = go.Figure()
    
    # Define colors for different elements
    colors = {
        'pred': 'red',
        'gt': 'blue', 
        'gt_action': 'green',
        'prev': 'yellow',
        'head': 'purple'
    }
    
    # Helper function to add coordinate frame
    def add_coordinate_frame(pos, name, color, size=0.05):
        # Add position marker
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='markers',
            name=f"{name}",
            marker=dict(size=8, color=color)
        ))
        
        # Add coordinate axes
        axis_colors = ['red', 'green', 'blue']  # x, y, z
        for i, ax_color in enumerate(axis_colors):
            axis_end = pos.copy()
            axis_end[i] += size
            fig.add_trace(go.Scatter3d(
                x=[pos[0], axis_end[0]],
                y=[pos[1], axis_end[1]],
                z=[pos[2], axis_end[2]],
                mode='lines',
                name=f"{name}_{['x', 'y', 'z'][i]}",
                line=dict(color=ax_color, width=3),
                showlegend=False
            ))
    
    # Add initial frame data
    if len(collect_3d_points) > 0:
        first_frame_idx = 0
        
        # Add predicted points
        pred_points = collect_3d_points[first_frame_idx].reshape(-1, 3)
        fig.add_trace(go.Scatter3d(
            x=pred_points[:, 0], y=pred_points[:, 1], z=pred_points[:, 2],
            mode='markers',
            name='Predicted Points',
            marker=dict(size=6, color=colors['pred'])
        ))
        
        # Add GT points
        gt_points = np.array(collect_3d_gt_points[first_frame_idx]).reshape(-1, 3)
        fig.add_trace(go.Scatter3d(
            x=gt_points[:, 0], y=gt_points[:, 1], z=gt_points[:, 2],
            mode='markers',
            name='Ground Truth Points',
            marker=dict(size=6, color=colors['gt'])
        ))
        
        # Add GT action points
        gt_action_points = collect_3d_gt_action_points[first_frame_idx].reshape(-1, 3)
        fig.add_trace(go.Scatter3d(
            x=gt_action_points[:, 0], y=gt_action_points[:, 1], z=gt_action_points[:, 2],
            mode='markers',
            name='GT Action Points',
            marker=dict(size=6, color=colors['gt_action'])
        ))
        
        # Add previous points
        prev_points = np.array(collect_3d_prev_points[first_frame_idx]).reshape(-1, 3)
        fig.add_trace(go.Scatter3d(
            x=prev_points[:, 0], y=prev_points[:, 1], z=prev_points[:, 2],
            mode='markers',
            name='Previous Points',
            marker=dict(size=6, color=colors['prev'])
        ))
        
        # Add head point
        head_point = collect_head_points[first_frame_idx]
        fig.add_trace(go.Scatter3d(
            x=[head_point[0]], y=[head_point[1]], z=[head_point[2]],
            mode='markers',
            name='Head',
            marker=dict(size=10, color=colors['head'])
        ))
    
    # Step 3: Update layout
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(title='X (meters)'),
            yaxis=dict(title='Y (meters)'),
            zaxis=dict(title='Z (meters)')
        ),
        title=f"3D Prediction Visualization - {dataset_name}",
        showlegend=True,
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {
                        'frame': {'duration': 100, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'type': 'buttons',
            'direction': 'left',
            'showactive': True,
            'x': 0.1,
            'y': 0.9
        }],
        sliders=[{
            'currentvalue': {'prefix': 'Frame: '},
            'pad': {'t': 50},
            'len': 0.9,
            'x': 0.1,
            'xanchor': 'left',
            'y': 0,
            'yanchor': 'top',
            'steps': [{
                'args': [[str(i)], {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }],
                'label': str(i),
                'method': 'animate'
            } for i in range(len(collect_3d_points))]
        }]
    )
    
    # Step 4: Collect animation frames
    print("Creating animation frames...")
    fig_frames = []
    
    for i in range(len(collect_3d_points)):
        frame_traces = []
        
        # Add predicted points for this frame
        pred_points = collect_3d_points[i].reshape(-1, 3)
        frame_traces.append(go.Scatter3d(
            x=pred_points[:, 0], y=pred_points[:, 1], z=pred_points[:, 2],
            mode='markers',
            marker=dict(size=6, color=colors['pred'])
        ))
        
        # Add GT points for this frame
        gt_points = np.array(collect_3d_gt_points[i]).reshape(-1, 3)
        frame_traces.append(go.Scatter3d(
            x=gt_points[:, 0], y=gt_points[:, 1], z=gt_points[:, 2],
            mode='markers',
            marker=dict(size=6, color=colors['gt'])
        ))
        
        # Add GT action points for this frame
        gt_action_points = collect_3d_gt_action_points[i].reshape(-1, 3)
        frame_traces.append(go.Scatter3d(
            x=gt_action_points[:, 0], y=gt_action_points[:, 1], z=gt_action_points[:, 2],
            mode='markers',
            marker=dict(size=6, color=colors['gt_action'])
        ))
        
        # Add previous points for this frame
        prev_points = np.array(collect_3d_prev_points[i]).reshape(-1, 3)
        frame_traces.append(go.Scatter3d(
            x=prev_points[:, 0], y=prev_points[:, 1], z=prev_points[:, 2],
            mode='markers',
            marker=dict(size=6, color=colors['prev'])
        ))
        
        # Add head point for this frame
        head_point = collect_head_points[i]
        frame_traces.append(go.Scatter3d(
            x=[head_point[0]], y=[head_point[1]], z=[head_point[2]],
            mode='markers',
            marker=dict(size=10, color=colors['head'])
        ))
        
        # Add trajectory lines for better visualization
        if i > 0:
            # Trajectory for predicted points
            pred_traj = collect_3d_points[max(0, i-5):i+1].reshape(-1, 3)
            frame_traces.append(go.Scatter3d(
                x=pred_traj[:, 0], y=pred_traj[:, 1], z=pred_traj[:, 2],
                mode='lines',
                line=dict(color=colors['pred'], width=2),
                opacity=0.6
            ))
            
            # Trajectory for GT points
            gt_traj = np.array(collect_3d_gt_points[max(0, i-5):i+1]).reshape(-1, 3)
            frame_traces.append(go.Scatter3d(
                x=gt_traj[:, 0], y=gt_traj[:, 1], z=gt_traj[:, 2],
                mode='lines',
                line=dict(color=colors['gt'], width=2),
                opacity=0.6
            ))
        
        fig_frames.append(go.Frame(data=frame_traces, name=str(i)))
    
    # Assign frames to figure
    fig.frames = fig_frames
    
    print(f"Created {len(fig_frames)} animation frames")
    
    # Show the figure and save
    fig.show()
    
    # Save as HTML
    output_filename = f"3d_prediction_animation_{dataset_name}.html"
    fig.write_html(output_filename)
    print(f"3D animation saved as {output_filename}")
    
    return fig, collect_3d_points, collect_3d_gt_points

# Example usage (you would call this instead of render_pred):
# render_3d_pred(collect_batch, gt_actions, pred_actions, dataset_name, action_chunk_size=1)



# def render_pred_wrist_translation(collect_batch, pred_actions, frame='world', fps=30):
#     object_lib_path = './hot3d/hot3d/dataset/assets/'
#     mano_path = './hot3d/mano_v1_2/models/'
#     object_library, mano_model = load_asset(object_lib_path, mano_path)
    


#     for batch , p_a in zip(collect_batch, pred_actions):
#         cur_state = batch['state'][0]
#         #print(cur_state, 'jjjj',p_a)
        
#         next_state = cur_state[0] + unnormalize(p_a.cpu().numpy(),'hot3d')

#         next_state_gt = cur_state[0] + batch['gt_action'].cpu().numpy() /10.0

#         #print(next_state)
#         next_state = next_state.reshape(2, -1)
#         #print(next_state)
#         sequence_name = batch['sequence_name'][0].decode()
#         sequence_path = os.path.join('./hot3d/hot3d/dataset/', sequence_name)
#         timestamp_ns = batch['timestamp_ns'][0]

#         next_timestamp_ns = batch['next_timestamp_ns'][0]

#         hot3d_data_func = HOT3D_data_func(sequence_path, object_library, mano_model)  

#         #next_timestamp_ns = hot3d_data_func.get_next_timestamp(timestamp_ns, fps)

#         next_rgb = hot3d_data_func.get_rgb_camera_data(next_timestamp_ns)
#         print('frame', frame)
#         point_2d_list = hot3d_data_func.transform_3d_to_2d(next_timestamp_ns, next_state, frame)

#         cur_hand_data = hot3d_data_func.get_hand_pose_data(timestamp_ns)

#         cur_hand_data_2d = hot3d_data_func.transform_3d_to_2d(timestamp_ns,  [cur_hand_data['left_wrist'], cur_hand_data['right_wrist']]) 
#         cur_hand_data_c = hot3d_data_func.transform_world_to_camera(timestamp_ns, [cur_hand_data['left_wrist'], cur_hand_data['right_wrist']])
#         cur_hand_data_c = np.concatenate(cur_hand_data_c)
#         print('cur hand', cur_hand_data_c)
#         print('cur_state', cur_state)

#         gt_hand_data = hot3d_data_func.get_hand_pose_data(next_timestamp_ns)
#         gt_hand_wrist = [gt_hand_data['left_wrist'], gt_hand_data['right_wrist']]
#         gt_2d_list = hot3d_data_func.transform_3d_to_2d(next_timestamp_ns, gt_hand_wrist) 

#         gt_list_camera = hot3d_data_func.transform_world_to_camera(next_timestamp_ns, gt_hand_wrist) 
#         print('=='*20)
#         print('pred')
#         print(next_state)
#         print('batch gt')
#         print(next_state_gt)
#         print('hot3d gt')
#         gt_list_camera = np.concatenate(gt_list_camera)
#         print(gt_list_camera)
#         print('data action')
#         print(gt_list_camera - cur_hand_data_c)
#         print('detoken action')
#         print(batch['gt_action'].cpu().numpy())

#         print('=='*20)

#         plt.clf()
#         plt.figure()
#         plt.imshow(next_rgb['image'], interpolation="nearest")
#         plt.scatter(
#             x=[x[0] for x in gt_2d_list if x is not None],
#             y=[x[1] for x in gt_2d_list if x is not None],
#             s=30,
#             c="b",
#         )

#         for a, b in zip(cur_hand_data_2d, gt_2d_list):
#             if a is not None and b is not None:
#                 plt.plot([a[0], b[0]], [a[1], b[1]], color='blue', linewidth=2)

#         plt.scatter(
#             x=[x[0] for x in point_2d_list if x is not None],
#             y=[x[1] for x in point_2d_list if x is not None],
#             s=30,
#             c="r",
#         )

#         if wandb.run is not None:
#             buffer = io.BytesIO()
#             plt.savefig(buffer, format="png")
#             buffer.seek(0)

#             pil_image = Image.open(buffer)
#             pil_image = pil_image.rotate(270, expand=True)  
#             # Log the image to WandB
#             wandb.log({"plot": wandb.Image(pil_image)})



#         #print('2D: ', point_2d)
#         #print('3D: ', next_state)

# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import numpy as np

# def render_3d_pred(collect_batch, gt_actions, pred_actions, dataset_name, action_chunk_size=1):
#     """
#     Render 3D visualization of predicted vs ground truth hand positions and camera positions
    
#     Args:
#         collect_batch: Batch of data containing states, extrinsics, etc.
#         gt_actions: Ground truth actions
#         pred_actions: Predicted actions
#         dataset_name: Name of the dataset
#         action_chunk_size: Size of action chunks to process
#     """
#     # Initialize data function based on dataset
#     if dataset_name == 'arctic':
#         arctic_data_func = ARCTIC_data_func()
#     elif dataset_name == 'h2o':
#         arctic_data_func = h2o_data_func()
#     elif dataset_name == 'hoi4d':
#         arctic_data_func = Hoi4d_data_func()
#     elif dataset_name == 'oakink2':
#         arctic_data_func = OakInk2DataFunc(dataset_root='/mnt/lab-tank/fang/OakInk2_data/', 
#                                          mano_path='/home/bidex/dex_openvla/OakInk2/asset/mano_v1_2')
#     elif dataset_name == 'taco':
#         arctic_data_func = TacoDataFunc(dataset_root='/mnt/lab-tank/fang/taco_data/', 
#                                       object_model_root='/mnt/lab-tank/fang/taco_data/object_models_released')
#     elif dataset_name == 'humanpolicy':
#         arctic_data_func = HumanpolicyDataFunc()
    
#     # Storage for 3D visualization data
#     frames_data = []
#     pred_sequence_name = ''
#     a_c_size = action_chunk_size
    
#     # Process each batch
#     for batch_idx, (batch, gt_a_list, p_a_list) in enumerate(zip(collect_batch, gt_actions, pred_actions)):
#         sequence_name = batch['sequence_name'][0]
#         if isinstance(sequence_name, bytes):
#             sequence_name = sequence_name.decode('utf-8')
#         elif not isinstance(sequence_name, str):
#             sequence_name = str(sequence_name)
            
#         timestamp = batch['timestamp_ns'][0]
#         extrinsic = batch['extrinsic'][0]
#         next_timestamp = batch['next_timestamp_ns'][0]
        
#         # Set sequence if changed
#         if sequence_name != pred_sequence_name:
#             pred_sequence_name = sequence_name
#             arctic_data_func.set_sequence(sequence_name, timestamp)
        
#         cur_state = batch['state'][0]
#         if isinstance(cur_state, torch.Tensor):
#             cur_state = cur_state.cpu().numpy()
        
#         # Get dataset action for ground truth
#         dataset_action = arctic_data_func.get_action(timestamp, next_timestamp)
#         a_idx = np.arange(a_c_size)
#         p_a_list = p_a_list[a_idx]
#         gt_action_list = gt_a_list[a_idx]
#         gt_action_list = dataset_action
        
#         # Process predicted trajectory
#         pred_world_positions = []
#         gt_world_positions = []
#         p_cur_state = cur_state.copy()
#         g_cur_state = cur_state.copy()
        
#         # Get camera positions in world coordinates
#         camera_pos_current = np.linalg.inv(extrinsic)[:3, 3]
#         next_extrinsic = arctic_data_func.get_extrinsic(next_timestamp)
#         camera_pos_next = np.linalg.inv(next_extrinsic)[:3, 3]
        
#         for step_idx, (p_a, gt_a) in enumerate(zip(p_a_list, gt_action_list)):
#             # Predicted trajectory
#             next_state_pred = p_cur_state + p_a.cpu().numpy()
#             p_cur_state = next_state_pred.copy()
            
#             # Convert to 3D points in camera frame
#             num_points = next_state_pred.shape[0] // 3
#             next_state_pred_3d = next_state_pred.reshape(num_points, 3)
            
#             # Transform to world coordinates
#             pred_world = transf_point_array_np(np.linalg.inv(extrinsic), next_state_pred_3d)
#             pred_world_positions.append(pred_world)
            
#             # Ground truth trajectory
#             gt_next_state = g_cur_state + gt_a.cpu().numpy()
#             g_cur_state = gt_next_state.copy()
            
#             # Convert to 3D points in camera frame
#             gt_next_state_3d = gt_next_state.reshape(num_points, 3)
            
#             # Transform to world coordinates
#             gt_world = transf_point_array_np(np.linalg.inv(extrinsic), gt_next_state_3d)
#             gt_world_positions.append(gt_world)
        
#         # Get initial hand positions in world coordinates
#         cur_state_3d = cur_state.reshape(num_points, 3)
#         initial_world_pos = transf_point_array_np(np.linalg.inv(extrinsic), cur_state_3d)
        
#         # Store frame data
#         frame_data = {
#             'batch_idx': batch_idx,
#             'timestamp': timestamp,
#             'next_timestamp': next_timestamp,
#             'sequence_name': sequence_name,
#             'initial_pos': initial_world_pos,
#             'pred_trajectory': pred_world_positions,
#             'gt_trajectory': gt_world_positions,
#             'camera_pos_current': camera_pos_current,
#             'camera_pos_next': camera_pos_next,
#             'extrinsic_current': extrinsic,
#             'extrinsic_next': next_extrinsic
#         }
#         frames_data.append(frame_data)
    
#     # Create 3D visualization
#     fig = go.Figure()


    
#     # Helper function to add coordinate frame
#     def add_coordinate_frame(pos, rotation_matrix, name, scale=0.05):
#         """Add coordinate frame axes to the plot"""
#         x_axis = pos + rotation_matrix[:, 0] * scale
#         y_axis = pos + rotation_matrix[:, 1] * scale
#         z_axis = pos + rotation_matrix[:, 2] * scale
        
#         # X-axis (red)
#         fig.add_trace(go.Scatter3d(
#             x=[pos[0], x_axis[0]], y=[pos[1], x_axis[1]], z=[pos[2], x_axis[2]],
#             mode='lines', line=dict(color='red', width=4),
#             name=f'{name}_x_axis', showlegend=False
#         ))
        
#         # Y-axis (green)
#         fig.add_trace(go.Scatter3d(
#             x=[pos[0], y_axis[0]], y=[pos[1], y_axis[1]], z=[pos[2], y_axis[2]],
#             mode='lines', line=dict(color='green', width=4),
#             name=f'{name}_y_axis', showlegend=False
#         ))
        
#         # Z-axis (blue)
#         fig.add_trace(go.Scatter3d(
#             x=[pos[0], z_axis[0]], y=[pos[1], z_axis[1]], z=[pos[2], z_axis[2]],
#             mode='lines', line=dict(color='blue', width=4),
#             name=f'{name}_z_axis', showlegend=False
#         ))
    
#     # Colors for different elements
#     colors = {
#         'initial': 'black',
#         'predicted': 'red',
#         'ground_truth': 'green',
#         'camera_current': 'blue',
#         'camera_next': 'cyan'
#     }
    
#     # Plot data for each frame
#     for i, frame_data in enumerate(frames_data):
#         # Initial hand positions
#         initial_pos = frame_data['initial_pos']
#         fig.add_trace(go.Scatter3d(
#             x=initial_pos[:, 0], y=initial_pos[:, 1], z=initial_pos[:, 2],
#             mode='markers', marker=dict(size=8, color=colors['initial']),
#             name=f'Initial_Hand_{i}' if i == 0 else f'Initial_Hand_{i}',
#             showlegend=(i == 0)
#         ))
        
#         # Predicted trajectory
#         for step_idx, pred_pos in enumerate(frame_data['pred_trajectory']):
#             alpha = 0.3 + 0.7 * (step_idx + 1) / len(frame_data['pred_trajectory'])  # Fade effect
#             fig.add_trace(go.Scatter3d(
#                 x=pred_pos[:, 0], y=pred_pos[:, 1], z=pred_pos[:, 2],
#                 mode='markers', marker=dict(size=6, color=colors['predicted'], opacity=alpha),
#                 name=f'Predicted_{i}' if step_idx == 0 and i == 0 else f'Predicted_{i}',
#                 showlegend=(step_idx == 0 and i == 0)
#             ))
        
#         # Ground truth trajectory
#         for step_idx, gt_pos in enumerate(frame_data['gt_trajectory']):
#             alpha = 0.3 + 0.7 * (step_idx + 1) / len(frame_data['gt_trajectory'])  # Fade effect
#             fig.add_trace(go.Scatter3d(
#                 x=gt_pos[:, 0], y=gt_pos[:, 1], z=gt_pos[:, 2],
#                 mode='markers', marker=dict(size=6, color=colors['ground_truth'], opacity=alpha),
#                 name=f'Ground_Truth_{i}' if step_idx == 0 and i == 0 else f'Ground_Truth_{i}',
#                 showlegend=(step_idx == 0 and i == 0)
#             ))
        
#         # Camera positions
#         camera_current = frame_data['camera_pos_current']
#         camera_next = frame_data['camera_pos_next']
        
#         # Current camera position
#         fig.add_trace(go.Scatter3d(
#             x=[camera_current[0]], y=[camera_current[1]], z=[camera_current[2]],
#             mode='markers', marker=dict(size=10, color=colors['camera_current'], symbol='diamond'),
#             name=f'Camera_Current_{i}' if i == 0 else f'Camera_Current_{i}',
#             showlegend=(i == 0)
#         ))
        
#         # Next camera position
#         fig.add_trace(go.Scatter3d(
#             x=[camera_next[0]], y=[camera_next[1]], z=[camera_next[2]],
#             mode='markers', marker=dict(size=10, color=colors['camera_next'], symbol='diamond'),
#             name=f'Camera_Next_{i}' if i == 0 else f'Camera_Next_{i}',
#             showlegend=(i == 0)
#         ))
        
#         # Camera coordinate frames
#         rot_current = frame_data['extrinsic_current'][:3, :3]
#         rot_next = frame_data['extrinsic_next'][:3, :3]
        
#         add_coordinate_frame(camera_current, rot_current.T, f'cam_current_{i}')
#         add_coordinate_frame(camera_next, rot_next.T, f'cam_next_{i}')
        
#         # Connect trajectories with lines
#         if len(frame_data['pred_trajectory']) > 1:
#             # Connect predicted trajectory
#             for step_idx in range(len(frame_data['pred_trajectory']) - 1):
#                 curr_pos = frame_data['pred_trajectory'][step_idx]
#                 next_pos = frame_data['pred_trajectory'][step_idx + 1]
#                 for point_idx in range(len(curr_pos)):
#                     fig.add_trace(go.Scatter3d(
#                         x=[curr_pos[point_idx, 0], next_pos[point_idx, 0]],
#                         y=[curr_pos[point_idx, 1], next_pos[point_idx, 1]],
#                         z=[curr_pos[point_idx, 2], next_pos[point_idx, 2]],
#                         mode='lines', line=dict(color=colors['predicted'], width=2),
#                         showlegend=False
#                     ))
            
#             # Connect ground truth trajectory
#             for step_idx in range(len(frame_data['gt_trajectory']) - 1):
#                 curr_pos = frame_data['gt_trajectory'][step_idx]
#                 next_pos = frame_data['gt_trajectory'][step_idx + 1]
#                 for point_idx in range(len(curr_pos)):
#                     fig.add_trace(go.Scatter3d(
#                         x=[curr_pos[point_idx, 0], next_pos[point_idx, 0]],
#                         y=[curr_pos[point_idx, 1], next_pos[point_idx, 1]],
#                         z=[curr_pos[point_idx, 2], next_pos[point_idx, 2]],
#                         mode='lines', line=dict(color=colors['ground_truth'], width=2),
#                         showlegend=False
#                     ))
        
#         # Connect initial to first predicted/gt positions
#         if frame_data['pred_trajectory']:
#             first_pred = frame_data['pred_trajectory'][0]
#             for point_idx in range(len(initial_pos)):
#                 fig.add_trace(go.Scatter3d(
#                     x=[initial_pos[point_idx, 0], first_pred[point_idx, 0]],
#                     y=[initial_pos[point_idx, 1], first_pred[point_idx, 1]],
#                     z=[initial_pos[point_idx, 2], first_pred[point_idx, 2]],
#                     mode='lines', line=dict(color=colors['predicted'], width=2, dash='dash'),
#                     showlegend=False
#                 ))
        
#         if frame_data['gt_trajectory']:
#             first_gt = frame_data['gt_trajectory'][0]
#             for point_idx in range(len(initial_pos)):
#                 fig.add_trace(go.Scatter3d(
#                     x=[initial_pos[point_idx, 0], first_gt[point_idx, 0]],
#                     y=[initial_pos[point_idx, 1], first_gt[point_idx, 1]],
#                     z=[initial_pos[point_idx, 2], first_gt[point_idx, 2]],
#                     mode='lines', line=dict(color=colors['ground_truth'], width=2, dash='dash'),
#                     showlegend=False
#                 ))
    
#     # Update layout
#     fig.update_layout(
#         title=f"3D Visualization - {dataset_name}",
#         scene=dict(
#             aspectmode='data',
#             camera=dict(
#                 up=dict(x=0, y=0, z=1),
#                 center=dict(x=0, y=0, z=0),
#                 eye=dict(x=1.5, y=1.5, z=1.5)
#             ),
#             xaxis_title="X (meters)",
#             yaxis_title="Y (meters)",
#             zaxis_title="Z (meters)"
#         ),
#         showlegend=True,
#         width=1000,
#         height=800
#     )
    
#     # Show the figure
#     fig.show()
    
#     # Save as HTML
#     fig.write_html(f"3d_visualization_{dataset_name}.html")
    
#     print(f"3D visualization saved as 3d_visualization_{dataset_name}.html")
#     print(f"Processed {len(frames_data)} frames")
    
#     return frames_data


# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import numpy as np
# import json
# import h5py
# import moviepy as mpy
# from matplotlib import pyplot as plt

