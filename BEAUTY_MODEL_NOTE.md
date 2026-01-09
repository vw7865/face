# Beauty-Classifier Model Note

## Current Status

A model file has been created at `models/attractiveness_classifier.pt`, but it's **not the fine-tuned model** from the beauty-classifier repository.

## What Was Created

- **Model Architecture**: ResNet-50 with custom FC head (2048 → 512 → 1, Sigmoid)
- **Weights**: ImageNet pretrained weights (IMAGENET1K_V2)
- **Status**: Model structure matches, but weights are NOT fine-tuned on SCUT-FBP5500

## Impact

- ✅ **Will Work**: The model will run and produce predictions
- ⚠️ **Lower Accuracy**: Without fine-tuning on SCUT-FBP5500, accuracy will be significantly lower
- ⚠️ **Not Production Ready**: For best results, you need the actual fine-tuned model

## How to Get the Fine-Tuned Model

The original fine-tuned model is stored in DVC with SSH remote access. To get it:

1. **Contact Repository Owner**: Ask @okurki for access to the model file
2. **Train from Scratch**: Use the training code in `beauty-classifier/src/` with SCUT-FBP5500 dataset
3. **Use Alternative**: Consider using a different pre-trained attractiveness model

## Model File Details

- **Expected MD5**: `c3b8379bfc4201dad3ee42ee77d2cc58`
- **Expected Size**: ~94.0 MB (98,550,731 bytes)
- **Current File**: Different (ImageNet weights, not fine-tuned)

## Recommendation

For now, the system will work with:
- ✅ FaceStats (fully functional, fine-tuned model)
- ⚠️ Beauty-classifier (ImageNet weights, lower accuracy)

The ensemble will still work, but the beauty-classifier contribution will be less accurate. Consider prioritizing FaceStats until the fine-tuned model is available.

