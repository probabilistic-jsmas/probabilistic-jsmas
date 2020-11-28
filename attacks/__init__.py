from .tf_jsma import attack as attack_jsma_targeted
from .tf_jsma_nt import attack as attack_jsma_non_targeted

from .tf_wjsma import attack as attack_wjsma_targeted
from .tf_wjsma_nt import attack as attack_wjsma_non_targeted

from .tf_tjsma import attack as attack_tjsma_targeted
from .tf_tjsma_nt import attack as attack_tjsma_non_targeted

from .tf_maximal_jsma import attack as attack_maximal_jsma
from .tf_maximal_wjsma import attack as attack_maximal_wjsma

from .generate_attacks import get_batch_indices, generate_attacks_targeted, generate_attacks_non_targeted, \
    generate_attacks_non_targeted_substitute
