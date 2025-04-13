from .bases import BaseImageDataset
import os.path as osp

class MOT20(BaseImageDataset):
    def __init__(self, root='', verbose=True, **kwargs):
        super(MOT20, self).__init__()
        self.dataset_dir = osp.join(root, 'MOT20/images')
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.train_label = osp.join(self.dataset_dir, 'train.txt')
        
        self._check_before_run()
        
        train = self._process_dir(self.train_label)
        query = []  # Placeholder, needs query.txt for evaluation
        gallery = []  # Placeholder, needs gallery.txt for evaluation
        
        if verbose:
            print("=> MOT20 loaded")
            self.print_dataset_statistics(train, query, gallery)
        
        self.train = train
        self.query = query
        self.gallery = gallery
        
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
    
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.train_label):
            raise RuntimeError("'{}' is not available".format(self.train_label))
    
    def _process_dir(self, label_file):
        data = []
        with open(label_file, 'r') as f:
            for line in f.readlines():
                img_path, pid, camid = line.strip().split()
                img_path = osp.join(self.dataset_dir, img_path)
                pid = int(pid)
                camid = int(camid)
                trackid = 0  # MOT20 doesn't use trackid, set to 0
                data.append((img_path, pid, camid, trackid))
        return data