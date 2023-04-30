import urllib
import io
from pathlib import Path
from PIL import Image
import io
import numpy as np
import math
import cv2
import segmentation_models as sm
import numpy as np

def get_pixel_size(latitude, longitude, zoom=20):
    size = 256 * (2 ** zoom)

    # resolution in degrees per pixel
    res_lat = math.cos(latitude * math.pi / 180.) * 360. / size
    res_lng = 360. / size

    res_lat_meter = res_lat * 111139
    res_long_meter = res_lng * 40075000 * math.cos(latitude * math.pi / 180.) / 360

    return res_lat_meter * res_long_meter

def get_aerial_image_from_lat_lon_as_numpy(latitude, longitude):
    """
    Retrieves an aerial image from bing maps centered at a given latitude
    and longitude and saves it as .jpg.

    The zoom level is set to 20 as default.

    Parameters
    ----------
    latitude : numeric

    longitude : numeric

    image_name : string
        Name of image file.

    """
    api_key = "AIzaSyBxSn-LCdXebcWHMahCYsQfNQowI-m1asc"
    horizontal = 512
    vertical = 512
    scale = 1
    zoom = 20

    params = urllib.parse.urlencode(
        {"center": f"{latitude},{longitude}", "zoom": zoom,
         "size": f"{horizontal}x{vertical}", "maptype": "satellite",
         "scale": scale, "key": api_key}
    )


    maps_url = "https://maps.googleapis.com/maps/api/staticmap"
    image_url = f"{maps_url}?{params}"
    image_file = urllib.request.urlopen(image_url).read()
    image = Image.open(io.BytesIO(image_file)).convert('RGB')
    image = np.array(image)

    pixel_size = get_pixel_size(latitude, longitude)
    # image_name = str(image_name) + '.jpg'

    return image, pixel_size


class Backend():
    label_classes_super = ['pvmodule', 'dormer', 'window', 'ladder', 'chimney', 'shadow',
                                                       'tree', 'unknown', "nothing"] #
    n_classes_superstructures = len(label_classes_super)
    label_classes_segment = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                             'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'flat', "background"]
    n_classes_segment = len(label_classes_segment)

    merged_labels = label_classes_segment + label_classes_super[:-1]
    activation = 'softmax'
    BACKBONE = "resnet34"

    segment_model = sm.Unet(BACKBONE, classes=n_classes_segment, activation=activation)
    superstructures_model = sm.Unet(BACKBONE, classes=n_classes_superstructures, activation=activation)

    weights_segment = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/gpu-instance-go/code/Users/ruslan.mammadov/RID/results/UNet_2_initial_segments.h5"
    segment_model.load_weights(weights_segment)

    weights_super = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/gpu-instance-go/code/Users/ruslan.mammadov/RID/results/UNet_2_initial.h5"
    superstructures_model.load_weights(weights_super)

    def process(self, longitude, latitude):
        # preprocess_input = sm.get_preprocessing(self.BACKBONE)
        image, pixel_size = get_aerial_image_from_lat_lon_as_numpy(longitude, latitude)
        image = np.expand_dims(image, axis=0)

        segment_mask = self.segment_model.predict(image)
        segment_vector = np.argmax(segment_mask.squeeze(), axis=2)

        super_mask = self.superstructures_model.predict(image)
        super_vector = np.argmax(super_mask.squeeze(), axis=2)

        super_vector_updated = super_vector + self.n_classes_segment
        background_class = self.n_classes_segment + self.n_classes_superstructures - 1
        merged_vector = super_vector_updated

        no_superstructures = super_vector_updated == background_class
        merged_vector[no_superstructures] = segment_vector[no_superstructures]

        number_not_flat = (merged_vector < 16).sum(axis=None)
        number_flat = (merged_vector == 16).sum(axis=None)
        pv_modules = (merged_vector == 18).sum(axis=None)

        details = {
            "flat_surface": number_flat * pixel_size,
            "not_flat_surface": number_not_flat * pixel_size,
            "pv_modules": pv_modules * pixel_size
        }

        return image[0], segment_vector, super_vector, merged_vector, details

    def get_labels(self):
        return self.label_classes_segment, self.label_classes_super, self.merged_labels


### Example ###



from matplotlib import pyplot as plt

backend = Backend()

label_classes_segment, label_classes_super, labels_merged = backend.get_labels()
image, segment_vector, super_vector, merged_vector, details = backend.process(48.399667, 11.977072)

## Show normal image
plt.imshow(image)
plt.show()

## Show merged labels
cmap = plt.cm.get_cmap('tab20', len(labels_merged))
plt.imshow(merged_vector, cmap=cmap, vmin=0, vmax=len(labels_merged) - 1)
plt.colorbar()
plt.legend(labels_merged, loc='lower right')
plt.show()
print({i: labels_merged[i] for i in range(len(labels_merged))})

## Show segment labels
cmap = plt.cm.get_cmap('tab20', len(label_classes_segment))
plt.imshow(segment_vector, cmap=cmap, vmin=0, vmax=len(label_classes_segment) - 1)
plt.colorbar()
plt.legend(label_classes_segment, loc='lower right')
plt.show()
print({i: label_classes_segment[i] for i in range(len(label_classes_segment))})

## Show superobject labels
cmap = plt.cm.get_cmap('tab20', len(label_classes_super))
plt.imshow(super_vector, cmap=cmap, vmin=0, vmax=len(label_classes_super) - 1)
plt.colorbar()
plt.legend(label_classes_super, loc='lower right')
plt.show()
print({i: label_classes_super[i] for i in range(len(label_classes_super))})

## Show details
print(f"Flat surface:\t\t{details['flat_surface']} m^2")
print(f"Not-flat surface:\t{details['not_flat_surface']} m^2")
print(f"PV modules:\t\t{details['pv_modules']} m^2")