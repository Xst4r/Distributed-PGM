def KL(A, B, thetap, thetaq, mu):
    return A - B + mu*(thetap - thetaq)