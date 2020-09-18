from imgaug import augmenters as iaa

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

test_augmentation = iaa.Sequential([
    iaa.Sequential([
        iaa.GammaContrast((0.5, 2.0))
    ])   
])

custom_augment = iaa.Sequential([
    iaa.OneOf([
        iaa.Flipud(1),
        iaa.Affine(rotate=90),
        iaa.Affine(rotate=180),
        iaa.Affine(rotate=270),
        iaa.Sequential([
            iaa.Flipud(1),
            iaa.Affine(rotate=90),
        ]),
        iaa.Sequential([
            iaa.Flipud(1),
            iaa.Affine(rotate=180),
        ]),
        iaa.Sequential([
            iaa.Flipud(1),
            iaa.Affine(rotate=270),
        ])
        # Use this only if the defect not belong to corner section
        # iaa.Affine(rotate=(-45,45))
    ]),
    sometimes(iaa.OneOf([
        iaa.GaussianBlur((0, 1.5)),
        iaa.AverageBlur(k=(1, 5)),
        iaa.MedianBlur(k=(1, 7)),
    ])
    ),
    sometimes(
        iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5))
    ),
    iaa.OneOf([
        iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
        iaa.GammaContrast((0.5, 2.0)) # improve or worsen the contrast
    ]),
        # Cannot use on dirty image
        # iaa.Emboss(alpha=(0.25, 1.0), strength=(0, 2.0))
        # Use this only if the defect have substaintal quality
        # iaa.SimplexNoiseAlpha(iaa.OneOf([
        #         iaa.EdgeDetect(alpha=(0.5, 1.0)),
        #         iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
        #     ])),
        # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        
        # Only work if you planned on train RGB image
        iaa.Add((-25, 25), per_channel=0.5),
    sometimes(
        iaa.FrequencyNoiseAlpha(
                    exponent=(-4, 0),
                    first=iaa.Multiply((0.5, 1.5), per_channel=True),
                    second=iaa.LinearContrast((0.5, 2.0))
                )
    ),
    iaa.OneOf([
        iaa.Dropout(p=(0.0001, 0.005)),
        iaa.CoarseDropout((0.001,0.005), size_percent=(0.25,0.75), per_channel=0.2)
    ])
])


complex_augmentation = iaa.Sequential([
    # iaa.CoarseDropout((0.001, 0.002), size_percent=0.03125),
    iaa.OneOf([
        iaa.Flipud(1),
        iaa.Affine(rotate=90),
        iaa.Affine(rotate=180),
        iaa.Affine(rotate=270),
        iaa.Sequential([
            iaa.Flipud(1),
            iaa.Affine(rotate=90),
        ]),
        iaa.Sequential([
            iaa.Flipud(1),
            iaa.Affine(rotate=180),
        ]),
        iaa.Sequential([
            iaa.Flipud(1),
            iaa.Affine(rotate=270),
        ])
    ]),
    iaa.SomeOf((0, 5),
        [
            sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
            iaa.OneOf([
                iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
            ]),
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
            # search either for all edges or for directed edges,
            # blend the result with the original image using a blobby mask
            iaa.SimplexNoiseAlpha(iaa.OneOf([
                iaa.EdgeDetect(alpha=(0.5, 1.0)),
                iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
            ])),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
            iaa.OneOf([
                iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
            ]),
            iaa.Invert(0.05, per_channel=True), # invert color channels
            iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
            iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
            # either change the brightness of the whole image (sometimes
            # per channel) or change the brightness of subareas
            iaa.OneOf([
                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                iaa.FrequencyNoiseAlpha(
                    exponent=(-4, 0),
                    first=iaa.Multiply((0.5, 1.5), per_channel=True),
                    second=iaa.LinearContrast((0.5, 2.0))
                )
            ]),
            iaa.OneOf([
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                iaa.GammaContrast((0.5, 2.0)) # improve or worsen the contrast
            ]),
            iaa.Grayscale(alpha=(0.0, 1.0)),
            sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
        ],
        random_order=True
    )
])

simple_augmentation = iaa.Sequential([
    iaa.OneOf([
        iaa.Flipud(1),
        iaa.Affine(rotate=90),
        iaa.Affine(rotate=180),
        iaa.Affine(rotate=270),
        iaa.Sequential([
            iaa.Flipud(1),
            iaa.Affine(rotate=90),
        ]),
        iaa.Sequential([
            iaa.Flipud(1),
            iaa.Affine(rotate=180),
        ]),
        iaa.Sequential([
            iaa.Flipud(1),
            iaa.Affine(rotate=270),
        ])
    ])
])