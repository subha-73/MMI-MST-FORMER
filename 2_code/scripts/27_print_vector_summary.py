import torch
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# --- PATH SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(os.path.join(ROOT_DIR, '2_code'))

from modules.spatial_encoder import SpatialEncoder
from modules.clinical_encoder import ClinicalEncoder
from modules.fusion_layer import FusionLayer
from modules.single_scale_transformer import SingleScaleTransformer
from modules.dataset_loader import GrapeDataset

DEVICE = torch.device("cpu")
MODEL_PATH = os.path.join(ROOT_DIR, '3_results', 'models', 'best_forecasting_model.pth')


def print_block(title, description):
    print("\n" + "="*90)
    print(title)
    print("-"*90)
    print(description)
    print("="*90)


def print_vector_grid(vector, values_per_row=8, precision=5):
    v = vector.detach().cpu().view(-1).numpy()
    fmt = f"{{:>{precision+6}.{precision}f}}"
    for i in range(0, len(v), values_per_row):
        row = v[i:i+values_per_row]
        print("  ".join(fmt.format(x) for x in row))


def run_single_step_audit():

    print("\n" + "="*90)
    print(" CLINICAL FORECASTING SYSTEM – INPUT / OUTPUT INSPECTION")
    print(" (Structured for Academic Evaluation)")
    print("="*90)

    # ---------------- LOAD DATA ----------------
    processed_dir = os.path.join(ROOT_DIR, '1_data', 'processed')
    tensor_path = os.path.join(processed_dir, 'grape_test_images1.pt')
    csv_path = os.path.join(processed_dir, 'grape_test.csv')

    dataset = GrapeDataset(tensor_path, csv_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    images, clin_data, targets = next(iter(loader))
    b, s, c, h, w = images.shape

    clin_dim = dataset.get_clinical_dim()
    out_dim = dataset.get_output_dim()

    spatial = SpatialEncoder(512)
    clinical = ClinicalEncoder(clin_dim, 128)
    fusion = FusionLayer(512, 128, 256)
    transformer = SingleScaleTransformer(256, out_dim)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    spatial.load_state_dict(checkpoint['spatial'])
    clinical.load_state_dict(checkpoint['clinical'])
    fusion.load_state_dict(checkpoint['fusion'])
    transformer.load_state_dict(checkpoint['transformer'])

    spatial.eval(); clinical.eval(); fusion.eval(); transformer.eval()

    with torch.no_grad():

        # ---------------- INPUT CONSTRUCTION ----------------
        spatial_feats = spatial(images.view(-1, c, h, w))
        clinical_feats = clinical(clin_data.view(-1, clin_dim))

        spatial_seq = spatial_feats.view(b, s, -1)
        clinical_seq = clinical_feats.view(b, s, -1)

        fused_sequence = fusion(spatial_seq, clinical_seq)

        # ---------------- MODEL OUTPUT ----------------
        predictions = transformer(fused_sequence)

        Z_LF = fused_sequence[0, s-1]
        pred_vf = predictions[0, 0]
        true_vf = targets[0, 0]

        # ========================= INPUTS =========================

        print_block(
            "INPUT–1: LONGITUDINAL PATIENT FEATURE SEQUENCE",
            "This input consists of visit-wise fused representations derived from retinal images "
            "and clinical attributes. Each row corresponds to one historical visit."
        )
        print(f"Input Shape: {fused_sequence[0].shape}  (Visits × 256 features)")
        print("Sample (first visit – first 10 values):")
        print(np.round(fused_sequence[0, 0][:10].cpu().numpy(), 5))

        print_block(
            "INPUT–2: LAST VISIT LATENT REPRESENTATION (256-D)",
            "This vector summarizes the most recent patient condition and serves as the immediate "
            "context for forecasting the next visual field."
        )
        print(f"Vector Shape: {Z_LF.shape}")
        print(f"Value Range: Min={Z_LF.min():.4f}, Max={Z_LF.max():.4f}\n")
        print_vector_grid(Z_LF)

        # ========================= OUTPUTS =========================

        print_block(
            "OUTPUT–1: FORECASTED VISUAL FIELD (NEXT VISIT)",
            "This is the predicted visual field for the next clinical visit. "
            "Each value corresponds to a standard test location in the visual field exam."
        )
        print(f"Output Shape: {pred_vf.shape}  (61 test locations)")
        print(f"Value Range: Min={pred_vf.min():.4f}, Max={pred_vf.max():.4f}\n")
        print_vector_grid(pred_vf, values_per_row=10)

        print_block(
            "OUTPUT–2: ACTUAL VISUAL FIELD (GROUND TRUTH)",
            "These are the real visual field measurements recorded clinically. "
            "They are shown here only for comparison and evaluation purposes."
        )
        print(f"Output Shape: {true_vf.shape}  (61 test locations)")
        print(f"Value Range: Min={true_vf.min():.4f}, Max={true_vf.max():.4f}\n")
        print_vector_grid(true_vf, values_per_row=10)


if __name__ == "__main__":
    run_single_step_audit()
