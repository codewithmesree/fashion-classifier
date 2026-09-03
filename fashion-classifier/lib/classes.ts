// Must match the class_names order used when training the model in the notebook.
export const CLASS_NAMES = [
  "T-shirt/top",
  "Trouser",
  "Pullover",
  "Dress",
  "Coat",
  "Sandal",
  "Shirt",
  "Sneaker",
  "Bag",
  "Ankle boot",
] as const;

export type ClassName = (typeof CLASS_NAMES)[number];
