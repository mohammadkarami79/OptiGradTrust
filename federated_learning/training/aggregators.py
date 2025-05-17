import torch
import copy
import numpy as np
from collections import defaultdict, OrderedDict
from federated_learning.config import config
import logging

logger = logging.getLogger(__name__)

class BaseAggregator:
    """
    کلاس پایه برای تمام روش‌های تجمیع در یادگیری فدرال
    """
    
    def __init__(self):
        """
        مقداردهی اولیه کلاس پایه تجمیع‌کننده
        """
        self.state = {}  # حالت داخلی برای تجمیع‌کننده (برای روش‌هایی مانند FedAdam)
    
    def aggregate(self, model, client_gradients, client_weights=None, **kwargs):
        """
        تجمیع کلی گرادیان‌ها و بازگشت گرادیان تجمیع شده و تاریخچه یادگیری.
        این متد با متد train سرور سازگار است.
        
        Args:
            model (torch.nn.Module): مدل سراسری
            client_gradients (list): لیست گرادیان‌های کلاینت‌ها
            client_weights (list, optional): وزن هر کلاینت در تجمیع
            **kwargs: پارامترهای اضافی برای روش‌های خاص
            
        Returns:
            tuple: (aggregated_gradient, client_history)
                aggregated_gradient: گرادیان تجمیع شده
                client_history: تاریخچه یادگیری کلاینت‌ها (اختیاری)
        """
        # تبدیل گرادیان‌های کلاینت به ساختار مورد نیاز
        # اگر گرادیان‌ها به صورت تنسور هستند، به دیکشنری تبدیل می‌کنیم
        if client_gradients and isinstance(client_gradients[0], torch.Tensor):
            # تبدیل گرادیان‌های تنسوری به دیکشنری با کلید 'flattened'
            dict_gradients = []
            for grad in client_gradients:
                dict_gradients.append({'flattened': grad})
            client_gradients = dict_gradients
        
        # فراخوانی متد تجمیع گرادیان‌ها که در کلاس‌های فرزند پیاده‌سازی شده است
        aggregated_gradient = self.aggregate_gradients(client_gradients, client_weights, **kwargs)
        
        # تاریخچه کلاینت (می‌تواند در کلاس‌های فرزند تغییر کند)
        client_history = {'weights': client_weights}
        
        # اگر گرادیان به صورت دیکشنری باشد، آن را تخت می‌کنیم
        if isinstance(aggregated_gradient, dict) and 'flattened' in aggregated_gradient:
            aggregated_gradient = aggregated_gradient['flattened']
        elif isinstance(aggregated_gradient, dict) and model is not None:
            # تبدیل دیکشنری گرادیان به تنسور تخت برای سازگاری با متد _update_global_model
            flat_list = []
            for name, param in model.named_parameters():
                if name in aggregated_gradient:
                    flat_list.append(aggregated_gradient[name].view(-1))
                else:
                    # اگر پارامتر در گرادیان نباشد، گرادیان صفر اضافه می‌کنیم
                    flat_list.append(torch.zeros_like(param.view(-1)))
            
            if flat_list:
                aggregated_gradient = torch.cat(flat_list)
            else:
                aggregated_gradient = torch.tensor([])
                
        return aggregated_gradient, client_history
    
    def aggregate_gradients(self, client_gradients, client_weights=None, **kwargs):
        """
        تجمیع گرادیان‌های کلاینت‌ها
        
        Args:
            client_gradients (list): لیست گرادیان‌های کلاینت‌ها (هر کدام به صورت دیکشنری)
            client_weights (list, optional): وزن هر کلاینت در تجمیع. اگر None باشد، وزن‌ها یکسان فرض می‌شوند.
            **kwargs: پارامترهای اضافی برای روش‌های خاص
            
        Returns:
            dict: گرادیان تجمیع شده
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def update_model(self, model, aggregated_gradients, lr=None):
        """
        بروزرسانی مدل سراسری با استفاده از گرادیان‌های تجمیع شده
        
        Args:
            model (torch.nn.Module): مدل سراسری
            aggregated_gradients (dict): گرادیان‌های تجمیع شده
            lr (float, optional): نرخ یادگیری. اگر None باشد، از نرخ یادگیری تنظیم شده استفاده می‌شود.
        
        Returns:
            torch.nn.Module: مدل بروزرسانی شده
        """
        if lr is None:
            lr = config.LEARNING_RATE
            
        for name, param in model.named_parameters():
            if name in aggregated_gradients and param.requires_grad:
                param.data = param.data - lr * aggregated_gradients[name]
                
        return model


class FedAvgAggregator(BaseAggregator):
    """
    تجمیع‌کننده FedAvg: میانگین وزن‌دار گرادیان‌های کلاینت‌ها
    
    Reference:
        McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    """
    
    def aggregate_gradients(self, client_gradients, client_weights=None):
        """
        تجمیع گرادیان‌ها با روش FedAvg (میانگین وزن‌دار)
        """
        if not client_gradients:
            return {}
            
        # اگر وزن‌ها مشخص نشده باشند، وزن یکسان برای همه کلاینت‌ها در نظر می‌گیریم
        if client_weights is None:
            client_weights = [1.0 / len(client_gradients)] * len(client_gradients)
        else:
            # نرمال‌سازی وزن‌ها برای اطمینان از جمع 1
            weight_sum = sum(client_weights)
            if weight_sum > 0:
                client_weights = [w / weight_sum for w in client_weights]
            else:
                client_weights = [1.0 / len(client_gradients)] * len(client_gradients)
                
        # تجمیع گرادیان‌ها با میانگین وزن‌دار
        aggregated_gradients = {}
        
        # ساخت لیست تمام کلیدهای موجود در گرادیان‌های کلاینت‌ها
        all_keys = set()
        for client_grad in client_gradients:
            all_keys.update(client_grad.keys())
            
        # تجمیع گرادیان‌ها با میانگین وزن‌دار برای هر پارامتر
        for key in all_keys:
            weighted_grads = []
            for i, client_grad in enumerate(client_gradients):
                if key in client_grad:
                    weighted_grads.append(client_grad[key] * client_weights[i])
            
            if weighted_grads:
                aggregated_gradients[key] = torch.stack(weighted_grads).sum(dim=0)
                
        logger.debug(f"FedAvg aggregated {len(client_gradients)} client gradients with weights {client_weights}")
        return aggregated_gradients


class FedBnAggregator(FedAvgAggregator):
    """
    تجمیع‌کننده FedBN: میانگین وزن‌دار گرادیان‌ها بدون تجمیع لایه‌های BatchNorm
    
    Reference:
        Li et al., "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization"
    """
    
    def aggregate_gradients(self, client_gradients, client_weights=None):
        """
        تجمیع گرادیان‌ها با روش FedBN (بدون تجمیع لایه‌های BatchNorm)
        """
        # ابتدا تجمیع معمولی FedAvg را انجام می‌دهیم
        aggregated_gradients = super().aggregate_gradients(client_gradients, client_weights)
        
        # حذف گرادیان‌های مربوط به لایه‌های BatchNorm
        filtered_gradients = {}
        for key, value in aggregated_gradients.items():
            # اگر نام پارامتر حاوی bn نباشد یا مربوط به وزن‌های متحرک BatchNorm نباشد، آن را نگه می‌داریم
            # ResNet downsample.1 layers are also BatchNorm layers and should be filtered
            if (('bn' not in key.lower()) and 
                ('running_' not in key) and 
                ('num_batches_tracked' not in key) and
                ('downsample.1.' not in key)):
                filtered_gradients[key] = value
                
        logger.debug(f"FedBN filtered out BatchNorm parameters from {len(aggregated_gradients)} to {len(filtered_gradients)}")
        return filtered_gradients


class FedADMMAggregator(BaseAggregator):
    """
    تجمیع‌کننده FedADMM: استفاده از Alternating Direction Method of Multipliers برای تجمیع
    
    Reference:
        Wang et al., "ADMM for Efficient Deep Learning with Global Convergence"
    """
    
    def __init__(self):
        """
        مقداردهی اولیه تجمیع‌کننده FedADMM
        """
        super().__init__()
        self.state = {
            'dual_variables': None,  # متغیرهای dual (ضرب‌کننده‌های لاگرانژ)
            'z_variables': None,  # متغیرهای کمکی z
            'iteration': 0  # شمارنده تکرار
        }
        self.rho = config.FEDADMM_RHO  # پارامتر penalty
        self.sigma = config.FEDADMM_SIGMA  # نرخ بروزرسانی متغیرهای dual
        self.iterations = config.FEDADMM_ITERATIONS  # تعداد تکرارهای ADMM در هر راند
    
    def aggregate_gradients(self, client_gradients, client_weights=None):
        """
        تجمیع گرادیان‌ها با روش FedADMM
        """
        if not client_gradients:
            return {}
            
        # اگر وزن‌ها مشخص نشده باشند، وزن یکسان برای همه کلاینت‌ها در نظر می‌گیریم
        if client_weights is None:
            client_weights = [1.0 / len(client_gradients)] * len(client_gradients)
        else:
            # نرمال‌سازی وزن‌ها
            weight_sum = sum(client_weights)
            if weight_sum > 0:
                client_weights = [w / weight_sum for w in client_weights]
            else:
                client_weights = [1.0 / len(client_gradients)] * len(client_gradients)
                
        # اگر اولین بار است، متغیرهای dual را مقداردهی اولیه کنیم
        if self.state['dual_variables'] is None:
            self.state['dual_variables'] = {}
            
            # برای هر پارامتر یک متغیر dual صفر ایجاد می‌کنیم
            for client_grad in client_gradients:
                for key, value in client_grad.items():
                    if key not in self.state['dual_variables']:
                        self.state['dual_variables'][key] = torch.zeros_like(value)
        
        # متغیرهای z را مقداردهی اولیه کنیم (میانگین وزن‌دار گرادیان‌ها)
        fed_avg = FedAvgAggregator()
        z_variables = fed_avg.aggregate_gradients(client_gradients, client_weights)
        self.state['z_variables'] = z_variables
        
        # انجام تکرارهای ADMM
        for _ in range(self.iterations):
            # بروزرسانی x (گرادیان‌های کلاینت‌ها)
            new_client_gradients = []
            for i, client_grad in enumerate(client_gradients):
                new_client_grad = {}
                for key, value in client_grad.items():
                    if key in self.state['z_variables'] and key in self.state['dual_variables']:
                        # x = argmin ||x - z + u||^2 + (ρ/2)||x - z||^2
                        new_value = (value + self.state['z_variables'][key] - 
                                    self.state['dual_variables'][key]) / (1 + self.rho)
                        new_client_grad[key] = new_value
                    else:
                        new_client_grad[key] = value
                new_client_gradients.append(new_client_grad)
                
            # بروزرسانی z (متغیر سراسری)
            z_old = self.state['z_variables'].copy()
            self.state['z_variables'] = fed_avg.aggregate_gradients(new_client_gradients, client_weights)
            
            # بروزرسانی متغیرهای dual (u)
            for key in self.state['dual_variables'].keys():
                if key in self.state['z_variables']:
                    # u = u + σ(x - z)
                    dual_update = 0
                    for i, client_grad in enumerate(new_client_gradients):
                        if key in client_grad:
                            dual_update += client_weights[i] * (client_grad[key] - self.state['z_variables'][key])
                    
                    self.state['dual_variables'][key] += self.sigma * dual_update
            
            # بررسی همگرایی
            z_diff_norm = 0
            for key in self.state['z_variables'].keys():
                if key in z_old:
                    z_diff_norm += torch.norm(self.state['z_variables'][key] - z_old[key]).item()
            
            if z_diff_norm < 1e-4:
                logger.debug(f"FedADMM converged in {_+1} iterations")
                break
                
        self.state['iteration'] += 1
        logger.debug(f"FedADMM completed iteration {self.state['iteration']}")
        
        return self.state['z_variables']


class FedAdamAggregator(BaseAggregator):
    """
    تجمیع‌کننده FedAdam: استفاده از الگوریتم Adam برای بروزرسانی مدل سراسری
    
    Reference:
        Reddi et al., "Adaptive Federated Optimization"
    """
    
    def __init__(self):
        """
        مقداردهی اولیه تجمیع‌کننده FedAdam
        """
        super().__init__()
        self.state = {
            'm': None,  # first moment estimate (میانگین متحرک گرادیان‌ها)
            'v': None,  # second moment estimate (میانگین متحرک مربع گرادیان‌ها)
            't': 0  # شمارنده زمان
        }
        self.beta1 = config.FEDADAM_BETA1
        self.beta2 = config.FEDADAM_BETA2
        self.epsilon = config.FEDADAM_EPSILON
        self.lr = config.FEDADAM_LR
    
    def aggregate_gradients(self, client_gradients, client_weights=None):
        """
        تجمیع گرادیان‌ها با روش FedAvg و ذخیره برای استفاده در بروزرسانی Adam
        """
        # ابتدا تجمیع معمولی FedAvg را انجام می‌دهیم
        fed_avg = FedAvgAggregator()
        aggregated_gradients = fed_avg.aggregate_gradients(client_gradients, client_weights)
        
        return aggregated_gradients
    
    def update_model(self, model, aggregated_gradients, lr=None):
        """
        بروزرسانی مدل سراسری با استفاده از الگوریتم Adam
        """
        lr = self.lr if lr is None else lr
        self.state['t'] += 1
        
        # مقداردهی اولیه m و v در اولین تکرار
        if self.state['m'] is None:
            self.state['m'] = {}
            self.state['v'] = {}
            for name, param in model.named_parameters():
                if param.requires_grad and name in aggregated_gradients:
                    self.state['m'][name] = torch.zeros_like(param.data)
                    self.state['v'][name] = torch.zeros_like(param.data)
        
        # بروزرسانی m و v و پارامترهای مدل با Adam
        for name, param in model.named_parameters():
            if name in aggregated_gradients and param.requires_grad:
                grad = aggregated_gradients[name]
                
                # بروزرسانی m و v
                self.state['m'][name] = self.beta1 * self.state['m'][name] + (1 - self.beta1) * grad
                self.state['v'][name] = self.beta2 * self.state['v'][name] + (1 - self.beta2) * (grad * grad)
                
                # تصحیح بایاس
                m_hat = self.state['m'][name] / (1 - self.beta1 ** self.state['t'])
                v_hat = self.state['v'][name] / (1 - self.beta2 ** self.state['t'])
                
                # بروزرسانی پارامترها
                param.data = param.data - lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
        
        logger.debug(f"FedAdam updated model with step size {lr} (iteration {self.state['t']})")
        return model


class FedNovaAggregator(BaseAggregator):
    """
    تجمیع‌کننده FedNova: نرمال‌سازی بروزرسانی‌های محلی براساس تعداد قدم‌های بهینه‌سازی
    
    Reference:
        Wang et al., "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization"
    """
    
    def __init__(self):
        """
        مقداردهی اولیه تجمیع‌کننده FedNova
        """
        super().__init__()
        self.normalize = config.FEDNOVA_NORMALIZE_UPDATES
    
    def aggregate_gradients(self, client_gradients, client_weights=None, client_steps=None):
        """
        تجمیع گرادیان‌ها با روش FedNova (نرمال‌سازی بر اساس تعداد قدم‌های محلی)
        
        Args:
            client_gradients (list): لیست گرادیان‌های کلاینت‌ها
            client_weights (list, optional): وزن هر کلاینت
            client_steps (list, optional): تعداد قدم‌های بهینه‌سازی محلی هر کلاینت
        """
        if not client_gradients:
            return {}
            
        # اگر وزن‌ها مشخص نشده باشند، وزن یکسان برای همه کلاینت‌ها در نظر می‌گیریم
        if client_weights is None:
            client_weights = [1.0 / len(client_gradients)] * len(client_gradients)
        else:
            # نرمال‌سازی وزن‌ها
            weight_sum = sum(client_weights)
            if weight_sum > 0:
                client_weights = [w / weight_sum for w in client_weights]
            else:
                client_weights = [1.0 / len(client_gradients)] * len(client_gradients)
        
        # اگر تعداد قدم‌ها مشخص نشده باشد، فرض می‌کنیم همه کلاینت‌ها یک قدم انجام داده‌اند
        if client_steps is None or not self.normalize:
            client_steps = [1] * len(client_gradients)
            
        # محاسبه مجموع وزن‌دار قدم‌ها
        tau_eff = sum(w * s for w, s in zip(client_weights, client_steps))
        
        # تجمیع گرادیان‌ها با نرمال‌سازی
        aggregated_gradients = {}
        
        # ساخت لیست تمام کلیدهای موجود در گرادیان‌های کلاینت‌ها
        all_keys = set()
        for client_grad in client_gradients:
            all_keys.update(client_grad.keys())
            
        # تجمیع گرادیان‌ها با FedNova برای هر پارامتر
        for key in all_keys:
            weighted_grads = []
            for i, client_grad in enumerate(client_gradients):
                if key in client_grad:
                    # نرمال‌سازی گرادیان براساس تعداد قدم‌ها
                    normalized_weight = client_weights[i] * client_steps[i] / tau_eff if tau_eff > 0 else client_weights[i]
                    weighted_grads.append(client_grad[key] * normalized_weight)
            
            if weighted_grads:
                aggregated_gradients[key] = torch.stack(weighted_grads).sum(dim=0)
                
        logger.debug(f"FedNova aggregated {len(client_gradients)} client gradients with normalized weights")
        return aggregated_gradients


class FedDWAAggregator(BaseAggregator):
    """
    تجمیع‌کننده FedDWA: وزن‌دهی پویا به کلاینت‌ها براساس عملکرد آنها
    
    Reference:
        Chai et al., "FedDWA: Federated Dynamic Weight Aggregation"
    """
    
    def __init__(self):
        """
        مقداردهی اولیه تجمیع‌کننده FedDWA
        """
        super().__init__()
        self.state = {
            'client_weights_history': {}  # تاریخچه وزن‌های کلاینت‌ها
        }
        self.weighting = config.FEDDWA_WEIGHTING  # روش وزن‌دهی ('loss', 'accuracy', یا 'gradient_norm')
        self.history_factor = config.FEDDWA_HISTORY_FACTOR  # ضریب تاریخچه (بین 0 و 1)
    
    def aggregate_gradients(self, client_gradients, client_metrics=None, client_ids=None):
        """
        تجمیع گرادیان‌ها با وزن‌دهی پویا براساس عملکرد کلاینت‌ها
        
        Args:
            client_gradients (list): لیست گرادیان‌های کلاینت‌ها
            client_metrics (dict, optional): دیکشنری شامل معیارهای هر کلاینت (دقت، loss و غیره)
            client_ids (list, optional): شناسه‌های کلاینت‌ها برای ردیابی تاریخچه
        """
        if not client_gradients:
            return {}
            
        num_clients = len(client_gradients)
        
        # اگر شناسه‌های کلاینت‌ها مشخص نشده باشند، از اندیس استفاده می‌کنیم
        if client_ids is None:
            client_ids = [f"client_{i}" for i in range(num_clients)]
            
        # اگر معیارها مشخص نشده باشند، وزن یکسان اختصاص می‌دهیم
        if client_metrics is None or not client_metrics:
            client_weights = [1.0 / num_clients] * num_clients
        else:
            # محاسبه وزن‌ها براساس معیار انتخاب شده
            if self.weighting == 'accuracy':
                # وزن بیشتر به کلاینت‌هایی که دقت بالاتری دارند
                metrics = [client_metrics[client_id].get('accuracy', 0) for client_id in client_ids]
            elif self.weighting == 'loss':
                # وزن بیشتر به کلاینت‌هایی که loss کمتری دارند
                metrics = [1.0 / (client_metrics[client_id].get('loss', 1) + 1e-5) for client_id in client_ids]
            elif self.weighting == 'gradient_norm':
                # وزن براساس نرم گرادیان‌ها (کلاینت‌هایی که تغییرات بزرگتری دارند)
                metrics = []
                for client_grad in client_gradients:
                    norm = 0
                    for param_grad in client_grad.values():
                        norm += torch.norm(param_grad).item() ** 2
                    metrics.append(np.sqrt(norm))
            else:
                metrics = [1.0] * num_clients
                
            # نرمال‌سازی معیارها به وزن‌ها
            total = sum(metrics)
            client_weights = [m / total if total > 0 else 1.0 / num_clients for m in metrics]
            
            # ترکیب وزن‌های جدید با تاریخچه
            for i, client_id in enumerate(client_ids):
                if client_id in self.state['client_weights_history']:
                    history_weight = self.state['client_weights_history'][client_id]
                    client_weights[i] = (self.history_factor * history_weight + 
                                        (1 - self.history_factor) * client_weights[i])
                
                # بروزرسانی تاریخچه
                self.state['client_weights_history'][client_id] = client_weights[i]
            
            # نرمال‌سازی نهایی وزن‌ها
            total = sum(client_weights)
            client_weights = [w / total if total > 0 else 1.0 / num_clients for w in client_weights]
        
        # استفاده از FedAvg با وزن‌های محاسبه شده
        fed_avg = FedAvgAggregator()
        aggregated_gradients = fed_avg.aggregate_gradients(client_gradients, client_weights)
        
        logger.debug(f"FedDWA aggregated {num_clients} client gradients with dynamic weights")
        return aggregated_gradients


class ScaffoldAggregator(BaseAggregator):
    """
    تجمیع‌کننده SCAFFOLD: استفاده از بردارهای کنترل برای کاهش واریانس بین کلاینت‌ها
    
    Reference:
        Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
    """
    
    def __init__(self):
        """
        مقداردهی اولیه تجمیع‌کننده SCAFFOLD
        """
        super().__init__()
        self.state = {
            'control_variate': None,  # بردار کنترل سراسری
            'client_control_variates': {}  # بردارهای کنترل کلاینت‌ها
        }
        self.control_lr = config.SCAFFOLD_CONTROL_LR  # نرخ بروزرسانی بردارهای کنترل
    
    def _initialize_control_variate(self, model):
        """
        مقداردهی اولیه بردار کنترل سراسری
        """
        control_variate = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                control_variate[name] = torch.zeros_like(param.data)
        return control_variate
    
    def get_client_control_variate(self, client_id, model):
        """
        دریافت بردار کنترل یک کلاینت
        """
        if self.state['control_variate'] is None:
            self.state['control_variate'] = self._initialize_control_variate(model)
            
        if client_id not in self.state['client_control_variates']:
            self.state['client_control_variates'][client_id] = copy.deepcopy(self.state['control_variate'])
            
        return self.state['client_control_variates'][client_id]
    
    def update_control_variates(self, client_id, delta_control_variate):
        """
        بروزرسانی بردارهای کنترل کلاینت و سراسری
        """
        # بروزرسانی بردار کنترل کلاینت
        for name, delta in delta_control_variate.items():
            if name in self.state['client_control_variates'][client_id]:
                self.state['client_control_variates'][client_id][name] += delta
                
        # بروزرسانی بردار کنترل سراسری
        for name in self.state['control_variate'].keys():
            if name in delta_control_variate:
                self.state['control_variate'][name] += self.control_lr * delta_control_variate[name]
    
    def aggregate_gradients(self, client_gradients, client_weights=None, client_delta_controls=None, client_ids=None):
        """
        تجمیع گرادیان‌ها با روش SCAFFOLD
        
        Args:
            client_gradients (list): لیست گرادیان‌های کلاینت‌ها
            client_weights (list, optional): وزن هر کلاینت
            client_delta_controls (list, optional): تغییرات بردارهای کنترل هر کلاینت
            client_ids (list, optional): شناسه‌های کلاینت‌ها
        """
        if not client_gradients:
            return {}
            
        # ابتدا تجمیع معمولی FedAvg را انجام می‌دهیم
        fed_avg = FedAvgAggregator()
        aggregated_gradients = fed_avg.aggregate_gradients(client_gradients, client_weights)
        
        # بروزرسانی بردارهای کنترل (اگر ارائه شده باشند)
        if client_delta_controls and client_ids:
            for i, client_id in enumerate(client_ids):
                if i < len(client_delta_controls):
                    self.update_control_variates(client_id, client_delta_controls[i])
        
        return aggregated_gradients
    
    def update_model(self, model, aggregated_gradients, lr=None):
        """
        بروزرسانی مدل سراسری با استفاده از گرادیان‌های تجمیع شده و بردار کنترل
        """
        if lr is None:
            lr = config.LEARNING_RATE
            
        # مقداردهی اولیه بردار کنترل سراسری اگر هنوز ایجاد نشده است
        if self.state['control_variate'] is None:
            self.state['control_variate'] = self._initialize_control_variate(model)
            
        # بروزرسانی پارامترهای مدل با گرادیان‌های تجمیع شده و اصلاح با بردار کنترل
        for name, param in model.named_parameters():
            if name in aggregated_gradients and param.requires_grad:
                if name in self.state['control_variate']:
                    # x_{t+1} = x_t - η(∇f(x_t) + c)
                    control_correction = self.state['control_variate'][name]
                    param.data = param.data - lr * (aggregated_gradients[name] + control_correction)
                else:
                    param.data = param.data - lr * aggregated_gradients[name]
                    
        logger.debug(f"SCAFFOLD updated model with step size {lr} and control correction")
        return model


class FedProxAggregator(FedAvgAggregator):
    """
    تجمیع‌کننده FedProx: استفاده از FedAvg با اضافه کردن یک عبارت نزدیکی به مدل سراسری
    
    Reference:
        Li et al., "Federated Optimization in Heterogeneous Networks"
    """
    
    def __init__(self):
        """
        مقداردهی اولیه تجمیع‌کننده FedProx
        """
        super().__init__()
        self.mu = config.FEDPROX_MU  # پارامتر μ برای عبارت نزدیکی
    
    def aggregate_gradients(self, client_gradients, client_weights=None):
        """
        تجمیع گرادیان‌ها با روش FedProx (همانند FedAvg)
        
        توجه: اعمال عبارت نزدیکی FedProx در سمت کلاینت انجام می‌شود (در training_utils.py)،
        بنابراین در اینجا فقط از تجمیع FedAvg استفاده می‌کنیم
        """
        # استفاده از همان روش FedAvg
        aggregated_gradients = super().aggregate_gradients(client_gradients, client_weights)
        logger.debug(f"FedProx aggregated with μ={self.mu}")
        return aggregated_gradients


class FedBnFedProxAggregator(FedBnAggregator):
    """
    تجمیع‌کننده FedBN_FedProx: ترکیبی از FedBN و FedProx
   استفاده از FedBN برای تجمیع (بدون تجمیع لایه‌های BatchNorm) و اضافه کردن عبارت نزدیکی FedProx
    """
    
    def __init__(self):
        """
        مقداردهی اولیه تجمیع‌کننده FedBN_FedProx
        """
        super().__init__()
        self.mu = config.FEDPROX_MU  # پارامتر μ برای عبارت نزدیکی FedProx
    
    def aggregate_gradients(self, client_gradients, client_weights=None):
        """
        تجمیع گرادیان‌ها با روش ترکیبی FedBN_FedProx
        
        توجه: اعمال عبارت نزدیکی FedProx در سمت کلاینت انجام می‌شود (در training_utils.py)،
        بنابراین در اینجا فقط از تجمیع FedBN استفاده می‌کنیم
        """
        # استفاده از همان روش FedBN
        aggregated_gradients = super().aggregate_gradients(client_gradients, client_weights)
        logger.debug(f"FedBN_FedProx aggregated with μ={self.mu}")
        return aggregated_gradients


def create_aggregator(method=None):
    """
    ایجاد تجمیع‌کننده مناسب براساس روش مشخص شده
    
    Args:
        method (str, optional): روش تجمیع. اگر None باشد، از روش پیش‌فرض در تنظیمات استفاده می‌شود.
    
    Returns:
        BaseAggregator: نمونه تجمیع‌کننده ایجاد شده
    """
    if method is None:
        method = config.AGGREGATION_METHOD
        
    method = method.lower()
    
    if method == 'fedavg':
        return FedAvgAggregator()
    elif method == 'fedbn':
        return FedBnAggregator()
    elif method == 'fedprox':
        return FedProxAggregator()
    elif method == 'fedbn_fedprox':
        return FedBnFedProxAggregator()
    elif method == 'fedadmm':
        return FedADMMAggregator()
    elif method == 'fedadam':
        return FedAdamAggregator()
    elif method == 'fednova':
        return FedNovaAggregator()
    elif method == 'feddwa':
        return FedDWAAggregator()
    elif method == 'scaffold':
        return ScaffoldAggregator()
    else:
        logger.warning(f"Unknown aggregation method '{method}', using FedAvg instead")
        return FedAvgAggregator() 