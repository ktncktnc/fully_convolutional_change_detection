def test(dset):
    net.eval()
    tot_loss = 0
    tot_count = 0
    tot_accurate = 0
    
    n = 2
    class_correct = list(0. for i in range(n))
    class_total = list(0. for i in range(n))
    class_accuracy = list(0. for i in range(n))
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for batch in train_loader:
        I1 = Variable(batch['I1'].float().cuda())
        I2 = Variable(batch['I2'].float().cuda())
        cm = torch.squeeze(Variable(batch['label'].cuda()))

        I1 = Variable(torch.unsqueeze(I1, 0).float()).cuda()
        I2 = Variable(torch.unsqueeze(I2, 0).float()).cuda()
        cm = Variable(torch.unsqueeze(1.0*cm,0).float()).cuda()


        output = net(I1, I2)
        loss = criterion(output, cm.long())
#         print(loss)
        tot_loss += loss.data * np.prod(cm.size())
        tot_count += np.prod(cm.size())

        _, predicted = torch.max(output.data, 1)

        c = (predicted.int() == cm.data.int())
        for i in range(c.size(1)):
            for j in range(c.size(2)):
                l = int(cm.data[0, i, j])
                class_correct[l] += c[0, i, j]
                class_total[l] += 1
                
        pr = (predicted.int() > 0).cpu().numpy()
        gt = (cm.data.int() > 0).cpu().numpy()
        
        tp += np.logical_and(pr, gt).sum()
        tn += np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
        fp += np.logical_and(pr, np.logical_not(gt)).sum()
        fn += np.logical_and(np.logical_not(pr), gt).sum()
        
    net_loss = tot_loss/tot_count
    net_accuracy = 100 * (tp + tn)/tot_count
    
    for i in range(n):
        class_accuracy[i] = 100 * class_correct[i] / max(class_total[i],0.00001)

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f_meas = 2 * prec * rec / (prec + rec)
    prec_nc = tn / (tn + fn)
    rec_nc = tn / (tn + fp)
    
    pr_rec = [prec, rec, f_meas, prec_nc, rec_nc]
        
    return net_loss, net_accuracy, class_accuracy, pr_rec
    
    