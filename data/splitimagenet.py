import os

import torch
from torchvision import datasets, transforms
from args import args

import torch.multiprocessing

import numpy as np

from copy import copy, deepcopy
from itertools import chain

torch.multiprocessing.set_sharing_strategy("file_system")


class SplitImageNet:
    def __init__(self):
        super(SplitImageNet, self).__init__()

        data_root = os.path.join(args.data, "imagenet")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        # Data loading code
        traindir = os.path.join(data_root, "train")
        valdir = os.path.join(data_root, "val")

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        train_datasets, val_datasets = self._construct_dataset_splits(
            train_dataset, val_dataset
        )

        self.train_loaders = []
        self.val_loaders = []

        for td, vd in zip(train_datasets, val_datasets):
            self.train_loaders.append(
                torch.utils.data.DataLoader(
                    td, batch_size=args.batch_size, shuffle=True, **kwargs
                )
            )

            self.val_loaders.append(
                torch.utils.data.DataLoader(
                    vd, batch_size=args.batch_size, shuffle=False, **kwargs
                )
            )

    def _construct_dataset_splits(self, train_dataset, val_dataset):
        print(f"==> Using seed {1} for ImageNet split")
        np.random.seed(1)

#        print("=> Generating split order with seed")
#        class_split_order = np.random.permutation(1000)

        print("=> Generating predictable split order")
# Freeze permutation to ensure conssitent results while being able to vary model initialization.
        class_split_order = np.array([114, 567, 745, 183, 412, 642, 297, 896, 651, 870, 168, 246, 759,
        827,  95, 226, 151, 840,  61, 278,  25, 159, 479, 231, 535,  16,
        77, 928, 826, 761, 121, 424, 284, 910, 178, 639, 532, 291, 197,
        34, 897, 792, 260, 513, 915, 473, 962, 312, 681, 237, 167, 757,
        696,  98, 691, 654, 683, 969, 556,   3, 557, 327, 505, 126, 289,
        982, 585, 702, 187, 496, 882, 674, 641, 384, 355, 565,  81,  80,
        666, 464, 808, 998, 252, 564, 591, 946, 125, 811, 448, 933, 485,
        863, 902, 202,  57, 474, 435, 990,  60, 275, 372, 815, 393, 800,
        744, 832, 615, 845, 307, 923, 709, 971, 287, 644, 102, 596, 682,
        420, 201, 385, 396, 413, 547,  12, 891, 116,  21, 620, 548, 687,
        605, 423, 546, 400, 152, 382, 123, 835, 261, 985, 267, 170, 227,
        243,   9, 224, 608,  24,  91, 244, 559, 421, 148, 786, 733, 427,
        308, 447, 511, 272, 154, 760, 672,  33, 119, 184, 833, 336, 769,
        799, 796, 636, 793,   5, 407, 225, 314, 839, 135, 310, 701,  63,
        741, 215, 737, 920, 698, 248, 203,  65, 403, 204, 935, 404, 653,
        853, 747, 173, 586,  48, 503, 468, 960, 703, 302, 952, 778,  52,
        162, 306, 972, 361, 587, 349, 255, 155, 631, 216, 771, 245, 519,
        322, 877, 319, 213, 378, 359, 205,  96, 951, 544, 607, 958, 108,
        256, 740, 582, 416, 841, 711, 137, 767,  38, 357,  90, 746, 929,
        961, 724, 459, 324, 238, 500, 619, 712, 223, 153, 489, 828, 765,
        270, 518, 199, 752, 161, 143, 253, 734, 803, 829, 847, 326, 616,
        868, 305, 859,  88, 892, 731, 401, 110, 487, 655, 860, 879, 570,
        783, 819, 283, 338, 282, 901, 341,  99, 456, 128, 940, 443, 706,
         39, 221, 713, 775,  47, 748, 735, 454, 742, 360, 780, 784, 331,
        211, 281, 886, 181, 350, 158, 402, 776, 276, 857, 117, 973, 727,
        991, 347, 346, 491, 483, 100, 617, 174, 966, 160, 524, 133,  17,
        139, 436, 943, 907, 949,  82, 371, 566, 191, 127, 195,  54,   2,
        198, 695, 947, 533, 689, 576, 369, 470, 938, 989, 874, 526, 721,
        730, 976, 528, 670,  22, 954, 507, 932,  92, 418,  78,  49, 983,
          8, 717, 699, 410, 516, 262, 738, 131, 773, 176, 669, 956, 575,
        550,  18, 814, 986, 156, 230, 536, 768, 217, 499, 574, 580, 997,
        430, 186, 573,  41, 510, 380, 806, 111, 710, 465, 823, 488, 390,
        172, 504, 540, 807, 508, 274, 667, 340, 207, 732, 269, 530, 273,
        208, 944,  32,  74, 439, 614, 460, 300,  19, 315,  94, 332, 354,
        581, 794, 950, 753, 816, 927, 212,  29, 838, 788, 817, 650, 180,
        146, 993, 764, 301, 671, 444, 873, 988, 517, 370, 392, 725, 445,
        309, 628, 140, 918, 843, 242,   1,  26, 129, 264,  71, 715, 635,
        321, 136, 389, 552, 254,  64, 395,  30, 630, 484, 862,  46, 222,
        978, 296, 495, 329, 394, 386, 656, 766, 493, 247, 209, 812, 171,
        979, 542, 294, 594, 458, 837, 498, 292, 668, 196, 749, 813, 381,
        411,  70, 980, 406, 851, 714, 210, 472, 995, 707, 720, 677, 649,
         45, 658, 343, 894, 643, 781, 930, 335, 782,  72, 864, 604, 105,
        553, 728, 467, 409, 633, 118, 387, 417, 551, 922, 189, 900, 520,
        953, 904, 351, 968, 419, 916, 831, 509, 572, 149, 673, 555, 848,
        795, 647, 164, 600, 797, 802, 538, 852, 664, 405, 686, 955, 250,
        478, 846, 229, 854, 132, 648, 834,  67, 442, 366, 241, 909, 779,
        685, 632, 188, 515, 101, 251, 561, 849, 589, 850, 214,  37, 362,
        492, 939,  75,  79,  28, 878, 428, 974, 583, 967, 948, 268, 688,
        452, 529, 867, 391, 763,  42, 115, 693, 466,  86, 109,  23, 239,
        317, 708, 919, 486, 462, 134, 446, 549, 652, 383, 122,  50, 977,
        560, 937, 895, 623, 629,  13, 432, 876,  83, 676, 981, 716, 175,
        665, 883, 240, 789, 376, 903, 893, 120, 471, 618, 913, 150, 352,
        626, 506, 539, 719, 723,  58, 534, 177, 220, 791, 625, 597, 558,
         35, 908, 758, 906, 762, 348, 569, 805,  44, 234, 342, 450, 637,
        157, 285, 936, 541, 739, 772, 325, 790, 798, 694, 842, 192, 729,
        679, 236, 690, 659, 645, 638, 316, 697, 521, 265, 169,  73, 182,
        646, 612, 855, 890, 494, 463, 232, 422, 363, 999, 593, 754, 692,
        801, 856, 822,   4, 449, 844,  27, 339, 905, 571, 660, 675, 774,
        965, 914, 945, 368, 662, 964, 106,  69, 963, 563, 311, 165, 434,
        480,  14, 545, 263, 634, 437, 959, 233, 408, 103,  59, 512,   7,
        185, 130, 277, 606, 590, 502, 917, 258, 228, 578, 704, 777,   0,
        266, 736, 290,  11, 657, 598, 885, 871, 562, 820, 433, 414, 179,
        678, 304, 624, 145, 477, 455, 865, 875, 888,  76, 295, 601, 344,
         56, 825, 194, 809, 755, 622, 398,  51, 531, 163, 987, 568, 751,
        770, 858, 787, 705, 756, 374,  31,  85, 279, 293, 206, 364, 912,
        722, 501, 461, 166, 785, 611, 337, 588, 663, 320, 610,  40, 476,
        609, 303, 994, 869, 249, 280, 441, 527, 375, 112, 415, 141, 318,
        884, 358, 830, 726, 921, 377,  84,   6, 235, 200, 824, 298, 718,
        942, 356, 996, 113, 218, 975, 595, 138, 880, 330, 440, 898, 889,
        810, 970, 482, 514, 899, 388, 887, 680, 613,  10,  68, 584, 743,
        931, 984, 219,  53, 592, 451, 481, 881, 353, 104,  15, 836, 599,
         66, 397, 379,  93, 603, 934, 957, 373, 453, 333, 345, 684, 525,
        554,  43, 299, 821, 640,  87,  20, 925, 288, 365,  62,  89, 399,
        750, 866, 142, 661, 941, 579,  36,  55, 144, 323, 497, 313, 367,
        426, 147, 475, 543, 872, 926, 818, 537, 190, 621, 259,  97, 271,
        861, 429, 193, 431, 124, 627, 469, 924, 328, 523, 107, 804, 425,
        992, 602, 438, 457, 334, 257, 577, 286, 911, 490, 522, 700])

        print("=> Splitting train dataset")
        train_datasets = self._split_dataset(train_dataset, class_split_order)

        print("=> Splitting val dataset")
        val_datasets = self._split_dataset(val_dataset, class_split_order)

        return train_datasets, val_datasets

    def _split_dataset(self, dataset, class_split_order):
#        task_length = len(class_split_order) // args.num_tasks
        task_length = args.output_size

        # Used to map from random task_length classes in {0...1000} -> {0,1...task_length}
#        tiled_class_map = np.tile(np.arange(task_length), args.num_tasks)
        tiled_class_map = np.tile(np.arange(task_length), 1000//task_length)
        inv_class_split_order = np.argsort(class_split_order)
        class_map = tiled_class_map[inv_class_split_order]

        # Constructing class splits
        paths, targets = zip(*dataset.samples)

        paths = np.array(paths)
        targets = np.array(targets)

        print("==> Extracting per class paths")
        class_samples = [
            list(zip(paths[targets == c], class_map[targets[targets == c]]))
            for c in range(1000)
        ]

        datasets = []

        print(f"==> Splitting dataset into {1000//task_length} tasks")
        for i in range(0, 1000, task_length):
            task_classes = class_split_order[i : i + task_length]

            samples = []

            for c in task_classes:
                samples.append(class_samples[c])

            redataset = copy(dataset)
            redataset.samples = list(chain.from_iterable(samples))

            datasets.append(redataset)

        return datasets

    def update_task(self, i):
        self.train_loader = self.train_loaders[i]
        self.val_loader = self.val_loaders[i]
