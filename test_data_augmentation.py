import unittest
from data_augmentation import PolymerDataAugmentor

class TestPolymerDataAugmentor(unittest.TestCase):
    def setUp(self):
        self.augmentor = PolymerDataAugmentor()

    def test_random_substitution(self):
        original_smiles = "CC(=O)OC1=CC=CC=C1C(=C)C(=O)O"
        augmented_smiles = self.augmentor.random_substitution(original_smiles)
        self.assertNotEqual(original_smiles, augmented_smiles)

    def test_augment(self):
        smiles_list = ["CC", "C(C)C", "C1=CC=CC=C1"]
        augmented_smiles = self.augmentor.augment(smiles_list)
        self.assertEqual(len(augmented_smiles), len(smiles_list))
        for smiles in augmented_smiles:
            self.assertNotEqual(smiles, "")  # Ensure no empty strings are returned

if __name__ == '__main__':
    unittest.main()