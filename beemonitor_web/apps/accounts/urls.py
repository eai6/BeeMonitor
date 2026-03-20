from django.urls import path

from . import views

app_name = "accounts"

urlpatterns = [
    path("login/", views.LoginView.as_view(), name="login"),
    path("logout/", views.LogoutView.as_view(), name="logout"),
    path("register/", views.RegisterView.as_view(), name="register"),
    path("usage/", views.UsageDashboardView.as_view(), name="usage"),
    path("redeem-coupon/", views.RedeemCouponView.as_view(), name="redeem_coupon"),
    path("coupons/", views.CouponListView.as_view(), name="coupons"),
    path("coupons/create/", views.CouponCreateView.as_view(), name="coupon_create"),
    path("coupons/<int:pk>/deactivate/", views.CouponDeactivateView.as_view(), name="coupon_deactivate"),
]
