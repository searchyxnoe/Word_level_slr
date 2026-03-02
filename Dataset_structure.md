# Before augmentation

videos/
├── validation/
│   ├── alert/          # 8 videos (1.mp4 to 8.mp4)
│   ├── careful/        # 8 videos (1.mp4 to 8.mp4)
│   ├── cheap/          # 8 videos (1.mp4 to 8.mp4)
│   ├── crazy/          # 8 videos (1.mp4 to 8.mp4)
│   ├── dangerous/      # 8 videos (1.mp4 to 8.mp4)
│   ├── decent/         # 8 videos (1.mp4 to 8.mp4)
│   ├── dumb/           # 8 videos (1.mp4 to 8.mp4)
│   ├── excited/        # 8 videos (1.mp4 to 8.mp4)
│   ├── extreme/        # 8 videos (1.mp4 to 8.mp4)
│   ├── fantastic/      # 8 videos (1.mp4 to 8.mp4)
│   ├── far/            # 8 videos (1.mp4 to 8.mp4)
│   ├── fearful/        # 8 videos (1.mp4 to 8.mp4)
│   ├── foreign/        # 8 videos (1.mp4 to 8.mp4)
│   ├── funny/          # 8 videos (1.mp4 to 8.mp4)
│   ├── good/           # 8 videos (1.mp4 to 8.mp4)
│   ├── healthy/        # 8 videos (1.mp4 to 8.mp4)
│   ├── heavy/          # 8 videos (1.mp4 to 8.mp4)
│   ├── important/      # 8 videos (1.mp4 to 8.mp4)
│   ├── intelligent/    # 8 videos (1.mp4 to 8.mp4)
│   ├── interesting/    # 8 videos (1.mp4 to 8.mp4)
│   ├── late/           # 8 videos (1.mp4 to 8.mp4)
│   ├── less/           # 8 videos (1.mp4 to 8.mp4)
│   ├── new/            # 8 videos (1.mp4 to 8.mp4)
│   ├── no/             # 8 videos (1.mp4 to 8.mp4)
│   ├── noisy/          # 8 videos (1.mp4 to 8.mp4)
│   ├── peaceful/       # 8 videos (1.mp4 to 8.mp4)
│   ├── quick/          # 8 videos (1.mp4 to 8.mp4)
│   ├── ready/          # 8 videos (1.mp4 to 8.mp4)
│   ├── secure/         # 8 videos (1.mp4 to 8.mp4)
│   ├── smart/          # 8 videos (1.mp4 to 8.mp4)
│   └── yes/            # 8 videos (1.mp4 to 8.mp4)
└── training/
    ├── alert/          # 48 videos (1.mp4 to 48.mp4)
    ├── careful/        # 48 videos (1.mp4 to 48.mp4)
    ├── cheap/          # 48 videos (1.mp4 to 48.mp4)
    ├── crazy/          # 48 videos (1.mp4 to 48.mp4)
    ├── dangerous/      # 48 videos (1.mp4 to 48.mp4)
    ├── decent/         # 48 videos (1.mp4 to 48.mp4)
    ├── dumb/           # 48 videos (1.mp4 to 48.mp4)
    ├── excited/        # 48 videos (1.mp4 to 48.mp4)
    ├── extreme/        # 48 videos (1.mp4 to 48.mp4)
    ├── fantastic/      # 48 videos (1.mp4 to 48.mp4)
    ├── far/            # 48 videos (1.mp4 to 48.mp4)
    ├── fearful/        # 48 videos (1.mp4 to 48.mp4)
    ├── foreign/        # 48 videos (1.mp4 to 48.mp4)
    ├── funny/          # 48 videos (1.mp4 to 48.mp4)
    ├── good/           # 48 videos (1.mp4 to 48.mp4)
    ├── healthy/        # 48 videos (1.mp4 to 48.mp4)
    ├── heavy/          # 48 videos (1.mp4 to 48.mp4)
    ├── important/      # 48 videos (1.mp4 to 48.mp4)
    ├── intelligent/    # 48 videos (1.mp4 to 48.mp4)
    ├── interesting/    # 48 videos (1.mp4 to 48.mp4)
    ├── late/           # 48 videos (1.mp4 to 48.mp4)
    ├── less/           # 48 videos (1.mp4 to 48.mp4)
    ├── new/            # 48 videos (1.mp4 to 48.mp4)
    ├── no/             # 48 videos (1.mp4 to 48.mp4)
    ├── noisy/          # 48 videos (1.mp4 to 48.mp4)
    ├── peaceful/       # 48 videos (1.mp4 to 48.mp4)
    ├── quick/          # 48 videos (1.mp4 to 48.mp4)
    ├── ready/          # 48 videos (1.mp4 to 48.mp4)
    ├── secure/         # 48 videos (1.mp4 to 48.mp4)
    ├── smart/          # 48 videos (1.mp4 to 48.mp4)
    └── yes/            # 48 videos (1.mp4 to 48.mp4)

# After augmentation

videos/
├── validation/
│   ├── alert/
│   │   ├── 1.mp4
│   │   ├── 2.mp4
│   │   ├── ...
│   │   └── 8.mp4
│   ├── careful/
│   │   ├── 1.mp4
│   │   ├── 2.mp4
│   │   ├── ...
│   │   └── 8.mp4
│   ├── cheap/
│   │   ├── 1.mp4
│   │   ├── 2.mp4
│   │   ├── ...
│   │   └── 8.mp4
│   ├── ...
│   └── yes/
│       ├── 1.mp4
│       ├── 2.mp4
│       ├── ...
│       └── 8.mp4
└── training/
    ├── alert/
    │   ├── 1.mp4
    │   ├── 2.mp4
    │   ├── ...
    │   ├── 8.mp4
    │   ├── 9.mp4   # Augmented version of 1.mp4
    │   ├── 10.mp4  # Augmented version of 2.mp4
    │   ├── ...
    │   └── 48.mp4  # Last augmented version
    ├── careful/
    │   ├── 1.mp4
    │   ├── 2.mp4
    │   ├── ...
    │   ├── 8.mp4
    │   ├── 9.mp4   # Augmented version of 1.mp4
    │   ├── 10.mp4  # Augmented version of 2.mp4
    │   ├── ...
    │   └── 48.mp4  # Last augmented version
    ├── cheap/
    │   ├── 1.mp4
    │   ├── 2.mp4
    │   ├── ...
    │   ├── 8.mp4
    │   ├── 9.mp4   # Augmented version of 1.mp4
    │   ├── 10.mp4  # Augmented version of 2.mp4
    │   ├── ...
    │   └── 48.mp4  # Last augmented version
    ├── ...
    └── yes/
        ├── 1.mp4
        ├── 2.mp4
        ├── ...
        ├── 8.mp4
        ├── 9.mp4   # Augmented version of 1.mp4
        ├── 10.mp4  # Augmented version of 2.mp4
        ├── ...
        └── 48.mp4  # Last augmented version
