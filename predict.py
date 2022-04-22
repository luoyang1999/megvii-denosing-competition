import megengine
import numpy as np
import tqdm

from model import Restormer_skffv3_ssa_share


def create_predict_bin(test_dir, result_dir, net, batch_size):
    content = open(test_dir, 'rb').read()

    samples_ref = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))

    fout = open(result_dir, 'wb')

    for i in tqdm.tqdm(range(0, len(samples_ref), batch_size)):
        i_end = min(i + batchsz, len(samples_ref))
        batch_inp = megengine.tensor(np.float32(samples_ref[i:i_end, None, :, :]) * np.float32(1 / 65536))
        pred = net(batch_inp).numpy()
        pred = (pred[:, 0, :, :] * 65536).clip(0, 65535).astype('uint16')
        fout.write(pred.tobytes())

    fout.close()
    print("Predicted result is saved at {}".format(result_dir))


if __name__ == '__main__':
    # load existing model
    net = Restormer_skffv3_ssa_share(shared_num=3, ffn_expansion_factor=2.18)
    pretrain_model = './weight.pth'
    model_info = megengine.load(pretrain_model)
    print('==> loading existing model:', pretrain_model)
    net.load_state_dict(model_info)

    # load test dataset
    test_dir = '../Data/burst_raw/competition_test_input.0.2.bin'
    result_dir = './result.bin'
    batchsz = 48

    create_predict_bin(test_dir, result_dir, net, batchsz)
