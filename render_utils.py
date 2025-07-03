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

def render_pred(collect_batch, pred_actions, dataset_name, action_chunk_size=1):
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
    elif dataset_name == 'oakink2':
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

    a_c_size = action_chunk_size # 1~4
    for batch, p_a_list in zip(collect_batch, pred_actions):


        sequence_name = batch['sequence_name'][0]
        if isinstance(sequence_name, bytes):
            sequence_name = sequence_name.decode('utf-8')
        elif not isinstance(sequence_name, str):
            sequence_name = str(sequence_name)
        timestamp = batch['timestamp_ns'][0]
        extrinsic = batch['extrinsic'][0]
        # print('extrinsic', extrinsic)
       
        
        next_timestamp = batch['next_timestamp_ns'][0]
        if sequence_name != pred_sequence_name:
            pred_sequence_name = sequence_name
            arctic_data_func.set_sequence(sequence_name, timestamp)

        cur_state = batch['state'][0]
        # print('cur_state: ', cur_state)
        
        #print(cur_state, 'jjjj',p_a)
        # print('lsklka batch: ', batch)
        # print('lsklka p_a: ', p_a)
        if isinstance(cur_state, torch.Tensor):
            cur_state = cur_state.cpu().numpy()
            
        # if len(p_a.shape) >= 2:
        #     p_a = p_a[0]    # only use the first action

        a_idx = np.arange(a_c_size)
        p_a_list = p_a_list[a_idx]
        gt_action_list = batch['gt_action'][a_idx]
        p_a_2d_list = []
        gt_a_2d_list = []
        p_cur_state = cur_state
        g_cur_state = cur_state
        for p_a, gt_a in zip(p_a_list, gt_action_list):

            pred_action = unnormalize(p_a.cpu().numpy(),dataset_name)

            next_state = p_cur_state + pred_action   #
            p_cur_state = next_state
            num_points = next_state.shape[0] // 3
            next_state = next_state.reshape(num_points, 3)  # camera frame at T=t
            next_state = transf_point_array_np(np.linalg.inv(extrinsic), next_state) #world frame
            next_extrinsic = arctic_data_func.get_extrinsic(next_timestamp)
            next_state = transf_point_array_np(next_extrinsic, next_state) #camera frame at T=t+1

            # from gt action
            gt_action = unnormalize(gt_a.cpu().numpy(), dataset_name)


            gt_action_next = g_cur_state + gt_action
            g_cur_state = gt_action_next
            num_points = gt_action_next.shape[0] // 3
            gt_action_next = gt_action_next.reshape(num_points, 3)
            gt_action_next = transf_point_array_np(np.linalg.inv(extrinsic), gt_action_next)
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

        # Plot lines only if both endpoints are in bounds
        for idx in range(len(render_hands)):
            # For gt line
            p0 = prev_points[idx]
            p1 = gt_points[idx]
            if (0 <= p0[0] < w and 0 <= p0[1] < h and 0 <= p1[0] < w and 0 <= p1[1] < h):
                x = np.array([p0[0], p1[0]])
                y = np.array([p0[1], p1[1]])
                plt.plot(x, y, color='green', linestyle='-', linewidth=2)

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


