# Rapport

Alla resultat finns i TensorFlow.

## TopDown-miljön (Status: Klar)

### PPO (Status: Semi-klar)
Den enda agenten jag hann (försöka) träna i den färdiga miljön (med checkpoints och randomized obstacles.) Agenten klarade inte av att komma i mål trots flera fixar, men klarade sig till checkpoint #2.

### TD3 (Status: Oklar)
Jag slog handskarna i disken med den här agenten ganska snabbt. Resultaten finns bifogade (de är före jag inkluderade randomized obstacles), och hittas på TensorFlow.

## GridBased (Status: Saknar randomized obstacles)
Eftersom jag inte ville få TopDown-miljön (och dess PPO-agent) att fungera saknar den här miljön randomized obstacles, och jag är inte säker på om någon av agenterna nedan har klarat av att lösa miljön. 

### PPO (Status: Oklar)
Jag försökte fixa TopDown PPO-agenten först, vilket ledde till att jag inte heller hade tid att slutföra den här agenten. 

### DQN (Status: Oklar)
Jag valde att lämna bort den här agenten tillsvidare, och fixa båda PPO agenterna.