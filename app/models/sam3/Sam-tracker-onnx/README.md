---
pipeline_tag: mask-generation
library_name: transformers.js
tags:
- sam3
base_model:
- facebook/sam3
---

## SAM3: Segment Anything with Concepts

### Usage (Transformers.js)

If you haven't already, you can install the [Transformers.js](https://huggingface.co/docs/transformers.js) JavaScript library from [NPM](https://www.npmjs.com/package/@huggingface/transformers) using:
```bash
npm i @huggingface/transformers
```

You can then use the model like this:
```js
import { Sam3TrackerModel, AutoProcessor, RawImage } from "@huggingface/transformers";

// Load model and processor
const model_id = "onnx-community/sam3-tracker-ONNX";
const model = await Sam3TrackerModel.from_pretrained(model_id);
const processor = await AutoProcessor.from_pretrained(model_id);

// Prepare image and input points/boxes
const img_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg";
const raw_image = await RawImage.read(img_url);

const input_points = [[[[500, 375]]]];
const input_labels = [[[1]]];
const input_boxes = undefined; // e.g., [[[75, 275, 1725, 850]]];

// Process inputs and perform mask generation
const inputs = await processor(raw_image, { input_points, input_labels, input_boxes });
const outputs = await model(inputs);

// Post-process masks
const masks = await processor.post_process_masks(outputs.pred_masks, inputs.original_sizes, inputs.reshaped_input_sizes);
// Tensor {
//   data: Uint8Array(6480000) [ 0, 0, 0, ... ],
//   type: 'bool',
//   dims: [ 1, 3, 1200, 1800 ],
//   size: 6480000
// }

const scores = outputs.iou_scores;
// Tensor {
//   data: Float32Array(3) [ 0.9313147068023682, 0.037515610456466675, 0.5128555297851562 ],
//   type: 'float32',
//   dims: [ 1, 1, 3 ],
//   size: 3
// }

// Visualize masks
const image = RawImage.fromTensor(masks[0][0].mul(255));
image.save("mask.png");
```

![mask](https://cdn-uploads.huggingface.co/production/uploads/61b253b7ac5ecaae3d1efe0c/OEYgs4z1FvbpgAAg1sosZ.png)