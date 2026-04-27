#include "gpgpu_driver.h"
#include <linux/atomic.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/fs.h>
#include <linux/interrupt.h>
#include <linux/io.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/pci.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/wait.h>

#define GPGPU_DRV_NAME "gpgpu_driver"

struct gpgpu_device {
  void __iomem *bar0;
  void __iomem *bar2;
  void __iomem *bar4;
  resource_size_t bar0_len;
  resource_size_t bar2_len;
  resource_size_t bar4_len;
  resource_size_t bar2_phys;
  dev_t devt;
  struct cdev cdev;
  struct class *class;
  struct device *device;
  struct mutex lock;
  wait_queue_head_t waitq;
  atomic_t irq_done;
  u32 irq_status;
  bool irq_requested;
  int irq;
};

static inline u32 gpgpu_bar0_read(const struct gpgpu_device *gdev,
                                  unsigned int offset, unsigned int size) {
  void __iomem *addr = gdev->bar0 + offset;

  switch (size) {
  case 1:
    return ioread8(addr);
  case 2:
    return ioread16(addr);
  case 4:
    return ioread32(addr);
  default:
    return 0;
  }
}

static inline void gpgpu_bar0_write(struct gpgpu_device *gdev,
                                    unsigned int offset, unsigned int size,
                                    u32 value) {
  void __iomem *addr = gdev->bar0 + offset;

  switch (size) {
  case 1:
    iowrite8(value, addr);
    break;
  case 2:
    iowrite16(value, addr);
    break;
  case 4:
    iowrite32(value, addr);
    break;
  }
}

static ssize_t gpgpu_chrdev_read(struct file *file, char __user *buf,
                                 size_t count, loff_t *ppos) {
  struct gpgpu_device *gdev = file->private_data;
  u32 value;
  ssize_t ret;

  if (mutex_lock_interruptible(&gdev->lock))
    return -ERESTARTSYS;
  if (*ppos >= gdev->bar0_len) {
    ret = 0;
    goto out;
  }
  if (count != 1 && count != 2 && count != 4) {
    ret = -EINVAL;
    goto out;
  }
  if (*ppos + count > gdev->bar0_len) {
    ret = -EINVAL;
    goto out;
  }

  value = gpgpu_bar0_read(gdev, *ppos, count);
  if (copy_to_user(buf, &value, count)) {
    ret = -EFAULT;
    goto out;
  }
  *ppos += count;
  ret = count;
out:
  mutex_unlock(&gdev->lock);
  return ret;
}

static ssize_t gpgpu_chrdev_write(struct file *file, const char __user *buf,
                                  size_t count, loff_t *ppos) {
  struct gpgpu_device *gdev = file->private_data;
  u32 value;
  ssize_t ret;

  if (mutex_lock_interruptible(&gdev->lock))
    return -ERESTARTSYS;
  if (*ppos >= gdev->bar0_len) {
    ret = -ENOSPC;
    goto out;
  }
  if (count != 1 && count != 2 && count != 4) {
    ret = -EINVAL;
    goto out;
  }
  if (*ppos + count > gdev->bar0_len) {
    ret = -EINVAL;
    goto out;
  }

  if (copy_from_user(&value, buf, count)) {
    ret = -EFAULT;
    goto out;
  }

  gpgpu_bar0_write(gdev, *ppos, count, value);
  *ppos += count;
  ret = count;
out:
  mutex_unlock(&gdev->lock);
  return ret;
}

static int gpgpu_chrdev_open(struct inode *inode, struct file *file) {
  struct gpgpu_device *gdev =
      container_of(inode->i_cdev, struct gpgpu_device, cdev);
  file->private_data = gdev;
  return 0;
}

static int gpgpu_chrdev_release(struct inode *inode, struct file *file) {
  return 0;
}

static irqreturn_t gpgpu_irq_handler(int irq, void *dev_id) {
  struct gpgpu_device *gdev = dev_id;
  u32 status;
  printk(KERN_DEBUG "GPGPU Driver: IRQ %d received\n", irq);
  status = gpgpu_bar0_read(gdev, GPGPU_REG_IRQ_STATUS, 4);
  if (!(status &
        (GPGPU_IRQ_KERNEL_DONE | GPGPU_IRQ_DMA_DONE | GPGPU_IRQ_ERROR)))
    return IRQ_NONE;

  gdev->irq_status = status;
  gpgpu_bar0_write(gdev, GPGPU_REG_IRQ_ACK, 4, status);

  atomic_set(&gdev->irq_done, 1);
  wake_up_interruptible(&gdev->waitq);
  return IRQ_HANDLED;
}

static long gpgpu_ioctl(struct file *file, unsigned int cmd,
                        unsigned long arg) {
  struct gpgpu_device *gdev = file->private_data;
  struct gpgpu_launch_params params;
  int ret;

  if (mutex_lock_interruptible(&gdev->lock))
    return -ERESTARTSYS;

  switch (cmd) {
  case GPGPU_IOCTL_LAUNCH:
    if (copy_from_user(&params, (void __user *)arg, sizeof(params))) {
      ret = -EFAULT;
      break;
    }

    atomic_set(&gdev->irq_done, 0);
    gdev->irq_status = 0;

    gpgpu_bar0_write(gdev, GPGPU_REG_GRID_DIM_X, 4, params.grid_dim[0]);
    gpgpu_bar0_write(gdev, GPGPU_REG_GRID_DIM_Y, 4, params.grid_dim[1]);
    gpgpu_bar0_write(gdev, GPGPU_REG_GRID_DIM_Z, 4, params.grid_dim[2]);
    gpgpu_bar0_write(gdev, GPGPU_REG_BLOCK_DIM_X, 4, params.block_dim[0]);
    gpgpu_bar0_write(gdev, GPGPU_REG_BLOCK_DIM_Y, 4, params.block_dim[1]);
    gpgpu_bar0_write(gdev, GPGPU_REG_BLOCK_DIM_Z, 4, params.block_dim[2]);
    gpgpu_bar0_write(gdev, GPGPU_REG_GLOBAL_CTRL, 4, params.global_ctrl);

    gpgpu_bar0_write(gdev, GPGPU_REG_DISPATCH, 4, 1);

    ret = wait_event_interruptible_timeout(
        gdev->waitq, atomic_read(&gdev->irq_done), 10 * HZ);
    if (ret == 0) {
      ret = -ETIMEDOUT;
      break;
    }
    if (ret < 0)
      break;

    if (!(gdev->irq_status & GPGPU_IRQ_KERNEL_DONE)) {
      ret = -EIO;
      break;
    }

    ret = 0;
    break;

  default:
    ret = -EINVAL;
    break;
  }

  mutex_unlock(&gdev->lock);
  return ret;
}

static int gpgpu_mmap(struct file *file, struct vm_area_struct *vma) {
  struct gpgpu_device *gdev = file->private_data;
  unsigned long vsize = vma->vm_end - vma->vm_start;
  unsigned long offset = vma->vm_pgoff << PAGE_SHIFT;
  unsigned long pfn;

  if (offset + vsize > gdev->bar2_len)
    return -EINVAL;

  vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
  pfn = (gdev->bar2_phys >> PAGE_SHIFT) + vma->vm_pgoff;
  if (remap_pfn_range(vma, vma->vm_start, pfn, vsize, vma->vm_page_prot))
    return -EAGAIN;

  return 0;
}

static const struct file_operations gpgpu_fops = {
    .owner = THIS_MODULE,
    .open = gpgpu_chrdev_open,
    .release = gpgpu_chrdev_release,
    .read = gpgpu_chrdev_read,
    .write = gpgpu_chrdev_write,
    .unlocked_ioctl = gpgpu_ioctl,
    .mmap = gpgpu_mmap,
    .llseek = default_llseek,
};

static int gpgpu_probe(struct pci_dev *pdev, const struct pci_device_id *id) {
  struct gpgpu_device *gdev;
  int ret;

  printk(KERN_INFO "GPGPU Driver: Found hardware at PCI %s\n", pci_name(pdev));

  ret = pci_enable_device(pdev);
  if (ret) {
    printk(KERN_ERR "GPGPU Driver: Failed to enable PCI device (%d)\n", ret);
    return ret;
  }

  pci_set_master(pdev);

  ret = pci_request_regions(pdev, GPGPU_DRV_NAME);
  if (ret) {
    printk(KERN_ERR "GPGPU Driver: Failed to request PCI regions (%d)\n", ret);
    goto disable_device;
  }

  gdev = kzalloc(sizeof(*gdev), GFP_KERNEL);
  if (!gdev) {
    ret = -ENOMEM;
    goto release_regions;
  }

  mutex_init(&gdev->lock);
  init_waitqueue_head(&gdev->waitq);
  atomic_set(&gdev->irq_done, 0);
  gdev->irq_status = 0;
  gdev->irq = -1;
  gdev->irq_requested = false;

  ret = alloc_chrdev_region(&gdev->devt, 0, 1, "gpgpu");
  if (ret)
    goto free_gdev;

  cdev_init(&gdev->cdev, &gpgpu_fops);
  gdev->cdev.owner = THIS_MODULE;
  ret = cdev_add(&gdev->cdev, gdev->devt, 1);
  if (ret)
    goto unregister_chrdev;

  gdev->class = class_create("gpgpu");
  if (IS_ERR(gdev->class)) {
    ret = PTR_ERR(gdev->class);
    goto del_cdev;
  }

  gdev->device =
      device_create(gdev->class, &pdev->dev, gdev->devt, gdev, "gpgpu0");
  if (IS_ERR(gdev->device)) {
    ret = PTR_ERR(gdev->device);
    goto destroy_class;
  }

  gdev->bar0 = pci_ioremap_bar(pdev, 0);
  if (!gdev->bar0) {
    ret = -EIO;
    goto destroy_device;
  }

  gdev->bar2 = pci_ioremap_bar(pdev, 2);
  if (!gdev->bar2) {
    ret = -EIO;
    goto unmap_bar0;
  }

  gdev->bar4 = pci_ioremap_bar(pdev, 4);
  if (!gdev->bar4) {
    ret = -EIO;
    goto unmap_bar2;
  }

  gdev->bar0_len = pci_resource_len(pdev, 0);
  gdev->bar2_len = pci_resource_len(pdev, 2);
  gdev->bar4_len = pci_resource_len(pdev, 4);
  gdev->bar2_phys = pci_resource_start(pdev, 2);

  ret = pci_alloc_irq_vectors(pdev, 1, 1, PCI_IRQ_MSI | PCI_IRQ_MSIX);
  if (ret < 0) {
    dev_warn(
        &pdev->dev,
        "GPGPU Driver: MSI/MSI-X vector allocation failed (%d), interrupt disabled\n",
        ret);
    gdev->irq = -1;
    ret = 0;
  } else {
    gdev->irq = pci_irq_vector(pdev, 0);
    if (gdev->irq < 0) {
      ret = -EINVAL;
      goto unmap_bar4;
    }

    ret = request_irq(gdev->irq, gpgpu_irq_handler, 0, "gpgpu_irq", gdev);
    if (ret) {
      dev_err(&pdev->dev, "GPGPU Driver: request_irq failed (%d)\n", ret);
      goto free_irq_vectors;
    }

    if (pdev->msix_enabled)
      dev_info(&pdev->dev, "GPGPU Driver: using MSI-X interrupt mode\n");
    else if (pdev->msi_enabled)
      dev_info(&pdev->dev, "GPGPU Driver: using MSI interrupt mode\n");
    else
      dev_warn(&pdev->dev,
               "GPGPU Driver: neither MSI-X nor MSI enabled unexpectedly\n");

    gpgpu_bar0_write(gdev, GPGPU_REG_IRQ_ENABLE, 4, GPGPU_IRQ_KERNEL_DONE);

    gdev->irq_requested = true;
    printk(KERN_INFO "GPGPU Driver: IRQ %d registered successfully\n",
           gdev->irq);
  }

  if (gpgpu_bar0_read(gdev, GPGPU_REG_DEV_ID, 4) != GPGPU_DEV_ID_VALUE ||
      gpgpu_bar0_read(gdev, GPGPU_REG_DEV_VERSION, 4) !=
          GPGPU_DEV_VERSION_VALUE) {
    printk(KERN_ERR "GPGPU Driver: Device ID or version mismatch\n");
    ret = -ENODEV;
    goto unmap_bar4;
  } else {
    printk(KERN_INFO "GPGPU Driver: Device ID and version verified\n");
  }

  printk(KERN_INFO "GPGPU Driver: BAR0 address: 0x%lx, length: 0x%lx bytes\n",
         (unsigned long)pci_resource_start(pdev, 0),
         (unsigned long)gdev->bar0_len);
  printk(KERN_INFO "GPGPU Driver: BAR2 address: 0x%lx, length: 0x%lx bytes\n",
         (unsigned long)pci_resource_start(pdev, 2),
         (unsigned long)gdev->bar2_len);
  printk(KERN_INFO "GPGPU Driver: BAR4 address: 0x%lx, length: 0x%lx bytes\n",
         (unsigned long)pci_resource_start(pdev, 4),
         (unsigned long)gdev->bar4_len);

  pci_set_drvdata(pdev, gdev);
  return 0;

unmap_bar4:
  pci_iounmap(pdev, gdev->bar4);
unmap_bar2:
  pci_iounmap(pdev, gdev->bar2);
unmap_bar0:
  pci_iounmap(pdev, gdev->bar0);
free_irq_vectors:
  if (gdev->irq_requested)
    free_irq(gdev->irq, gdev);
  pci_free_irq_vectors(pdev);
destroy_device:
  device_destroy(gdev->class, gdev->devt);
destroy_class:
  class_destroy(gdev->class);
del_cdev:
  cdev_del(&gdev->cdev);
unregister_chrdev:
  unregister_chrdev_region(gdev->devt, 1);
free_gdev:
  kfree(gdev);
release_regions:
  pci_release_regions(pdev);
disable_device:
  pci_disable_device(pdev);
  return ret;
}

static void gpgpu_remove(struct pci_dev *pdev) {
  struct gpgpu_device *gdev = pci_get_drvdata(pdev);

  if (gdev) {
    if (gdev->irq_requested)
      free_irq(gdev->irq, gdev);
    pci_free_irq_vectors(pdev);
    if (gdev->bar4)
      pci_iounmap(pdev, gdev->bar4);
    if (gdev->bar2)
      pci_iounmap(pdev, gdev->bar2);
    if (gdev->bar0)
      pci_iounmap(pdev, gdev->bar0);
    if (gdev->device)
      device_destroy(gdev->class, gdev->devt);
    if (gdev->class)
      class_destroy(gdev->class);
    cdev_del(&gdev->cdev);
    unregister_chrdev_region(gdev->devt, 1);
    kfree(gdev);
  }

  pci_release_regions(pdev);
  pci_disable_device(pdev);
  printk(KERN_INFO "GPGPU Driver: Device removed.\n");
}

static struct pci_device_id gpgpu_ids[] = {
    {PCI_DEVICE(GPGPU_VENDOR_ID, GPGPU_DEVICE_ID)},
    {0},
};
MODULE_DEVICE_TABLE(pci, gpgpu_ids);

static struct pci_driver gpgpu_pci_driver = {
    .name = GPGPU_DRV_NAME,
    .id_table = gpgpu_ids,
    .probe = gpgpu_probe,
    .remove = gpgpu_remove,
};

static int __init gpgpu_init(void) {
  return pci_register_driver(&gpgpu_pci_driver);
}

static void __exit gpgpu_exit(void) {
  pci_unregister_driver(&gpgpu_pci_driver);
}

module_init(gpgpu_init);
module_exit(gpgpu_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Han Zhang");
MODULE_DESCRIPTION("Minimal GPGPU Driver for QEMU Experiment");
