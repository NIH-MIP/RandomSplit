# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# September 2026
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import numpy as np
import torch
import scipy.linalg.lapack as la

# XXX:  Although... torch.ormqr exists too. Would it work with scipy's dgeqp3?
class _ormqr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, side, trans, a, tau, c):
        if a.device.type != "cpu" or tau.device != a.device or c.device != a.device:
            raise RuntimeError("Only CPU tensors are supported.")

        if a.dtype != tau.dtype or a.dtype != c.dtype:
            raise RuntimeError("All dtypes are expected to be the same.")

        if a.dtype == torch.float64:
            ormqr = la.dormqr
        elif a.dtype == torch.float32:
            ormqr = la.sormqr
        else:
            raise RuntimeError("Only torch.float32 or torch.float64 are supported.")

        ctx.side = side
        ctx.trans = trans
        ctx.save_for_backward(a, tau)

        np_a = a.numpy()
        np_tau = tau.numpy()
        np_c = c.numpy()

        if trans != "T" and trans != "N":
            raise RuntimeError(f"Unsupported 'trans' parameter value '{trans}'. Can only be 'N' or 'T'.") 

        if side not in ("L", "R"):
            raise RuntimeError(f"Unsupported 'side' parameter value '{side}'. Can only be 'L' or 'R'.")

        qc, work, info = ormqr(side=side, trans=trans, a=np_a, tau=np_tau, c=np_c, lwork=-1)
        lwork = work[0]

        assert info == 0

        qc, work, info = ormqr(side=side, trans=trans, a=np_a, tau=np_tau, c=np_c, lwork=lwork)

        assert info == 0

        return torch.from_numpy(qc).clone()

    @staticmethod
    def backward(ctx, dc):
        if any(ctx.needs_input_grad[:4]):
            raise RuntimeError("Only gradient computation is supported for the c parameter.")

        if not ctx.needs_input_grad[4]: # Nothing to do
            return None, None, None, None, None

        a, tau = ctx.saved_tensors
        side, trans = ctx.side, ctx.trans

        if dc.device != a.device:
            raise RuntimeError("c gradient is not on the same device as a.")

        if dc.dtype != a.dtype:
            raise RuntimeError("All dtypes are expected to be the same.")

        if a.dtype == torch.float64:
            ormqr = la.dormqr
        elif a.dtype == torch.float32:
            ormqr = la.sormqr
        else:
            raise RuntimeError("Only torch.float32 or torch.float64 are supported.")

        np_a = a.numpy()
        np_tau = tau.numpy()
        np_dc = dc.numpy()

        # Invert trans
        trans = "T" if trans == "N" else "N"

        qc, work, info = ormqr(side=side, trans=trans, a=np_a, tau=np_tau, c=np_dc, lwork=-1)
        lwork = work[0]

        assert info == 0

        qc, work, info = ormqr(side=side, trans=trans, a=np_a, tau=np_tau, c=np_dc, lwork=lwork)

        assert info == 0

        return None, None, None, None, torch.from_numpy(qc).clone()

def geqp3(a):
    if a.device.type != "cpu":
        raise RuntimeError("Only CPU tensors are supported.")

    if a.dtype == torch.float32:
        func = la.sgeqp3
    elif a.dtype == torch.float64:
        func = la.dgeqp3
    else:
        raise RuntimeError("Only torch.float32 or torch.float64 are supported.")

    if a.requires_grad:
        raise RuntimeError("This function does not support gradients.")

    if a.ndim != 2:
        raise RuntimeError("a is expected to be a 2D tensor.")

    np_a = a.numpy()

    h, jpvt, tau, work, info = func(np_a)

    assert info == 0

    h = torch.from_numpy(h).clone()
    jpvt = torch.from_numpy(jpvt).clone()
    tau = torch.from_numpy(tau).clone()

    return h, jpvt, tau

def ormqr(side, trans, a, tau, c):
    return _ormqr.apply(side, trans, a, tau, c)

